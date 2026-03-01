---
title: "SharpLeNet. DatabaseModelSaver: Когда моделей много, а порядок нужен"
date: 2026-03-01
description: "SharpLeNet. DatabaseModelSaver: Когда моделей много, а порядок нужен"
tags: ["SharpLeNet"]
categories: ["SharpLeNet"]
series: ["SharpLeNet"]
reading_time: 5
featured: true
---

Друзья, мы уже умеем сохранять модели в файлы — бинарные и JSON. Но что если у нас не одна-две модели, а сотни? Что если нужно хранить разные версии, искать по метаданным, делать сложные запросы? Файловая система тут не помощник — она плоский мир без индексов и связей.

На помощь приходят базы данных. Сегодня мы реализуем **DatabaseModelSaver** — сохранятор, который складывает модели в реляционную БД. С таблицами, связями, индексами и транзакциями. Всё по-взрослому.

## Архитектура: Два уровня абстракции

Наша система состоит из двух уровней:

1. **IDatabaseConnection** — абстракция над конкретной СУБД (SQLite, PostgreSQL, MySQL)
2. **DatabaseModelSaver** — реализация IModelSaver, работающая через этот интерфейс

Это позволяет легко менять базу данных, не трогая код сохранятора.

## IDatabaseConnection: Абстракция над БД

```csharp
public interface IDatabaseConnection : IDisposable
{
    void Open();
    void Close();
    int ExecuteNonQuery(string sql, params (string name, object value)[] parameters);
    IEnumerable<T> ExecuteQuery<T>(string sql, Func<IDataReader, T> mapper, params (string name, object value)[] parameters);
    T? ExecuteScalar<T>(string sql, params (string name, object value)[] parameters);
    void BeginTransaction();
    void CommitTransaction();
    void RollbackTransaction();
}
```

Это минимальный набор, достаточный для работы с любой реляционной БД:

- `Open/Close` — управление подключением
- `ExecuteNonQuery` — для INSERT, UPDATE, DELETE
- `ExecuteQuery` — для SELECT с маппингом результатов
- `ExecuteScalar` — для получения одного значения (например, last_insert_rowid)
- Транзакции — для атомарности операций

## SqliteConnection: Конкретная реализация

```csharp
public class SqliteConnection : IDatabaseConnection
{
    private readonly string _connectionString;
    private SqliteConnection? _connection;

    public SqliteConnection(string databasePath)
    {
        _connectionString = $"Data Source={databasePath}";
    }

    public void Open()
    {
        _connection = new SqliteConnection(_connectionString);
        _connection.Open();
    }

    // ... остальные методы ...
}
```

Здесь мы используем SQLite — легковесную встраиваемую БД, идеальную для хранения моделей. Файл базы данных можно легко копировать, версионировать, передавать коллегам.

## Схема базы данных: Три таблицы

Перед тем как сохранять модели, нужно создать структуру:

```csharp
private void InitializeDatabase()
{
    _connection.Open();
    try
    {
        // Таблица моделей
        _connection.ExecuteNonQuery(@"
            CREATE TABLE IF NOT EXISTS Models (
                Id INTEGER PRIMARY KEY AUTOINCREMENT,
                Identifier TEXT UNIQUE NOT NULL,
                CreatedAt DATETIME NOT NULL,
                Description TEXT,
                Version INTEGER NOT NULL DEFAULT 1,
                TotalLayers INTEGER NOT NULL DEFAULT 0,
                TotalParameters INTEGER NOT NULL DEFAULT 0
            )");
```

**Models** — метаданные о моделях:

- `Identifier` — уникальное имя модели (как в IModelSaver)
- `CreatedAt` — когда сохранили
- `Version` — версия (для будущих обновлений)
- `TotalLayers`, `TotalParameters` — для быстрой информации без загрузки

```csharp
// Таблица слоев
_connection.ExecuteNonQuery(@"
    CREATE TABLE IF NOT EXISTS Layers (
        Id INTEGER PRIMARY KEY AUTOINCREMENT,
        ModelId INTEGER NOT NULL,
        LayerIndex INTEGER NOT NULL,
        LayerType TEXT NOT NULL,

        -- Параметры линейного слоя
        InputSize INTEGER,
        OutputSize INTEGER,

        -- Параметры сверточного слоя
        InputChannels INTEGER,
        OutputChannels INTEGER,
        KernelSize INTEGER,
        Stride INTEGER,
        Padding INTEGER,

        FOREIGN KEY(ModelId) REFERENCES Models(Id) ON DELETE CASCADE,
        UNIQUE(ModelId, LayerIndex)
    )");
```

**Layers** — описание слоёв:

- `ModelId` — связь с моделью
- `LayerIndex` — порядковый номер слоя
- `LayerType` — тип слоя (LinearLayer, Conv2DLayer и т.д.)
- Параметры для разных типов слоёв (многие могут быть NULL)

```csharp
// Таблица тензоров
_connection.ExecuteNonQuery(@"
    CREATE TABLE IF NOT EXISTS Tensors (
        Id INTEGER PRIMARY KEY AUTOINCREMENT,
        LayerId INTEGER NOT NULL,
        ParameterIndex INTEGER NOT NULL,
        ShapeJson TEXT NOT NULL,
        DataBlob BLOB NOT NULL,
        DataLength INTEGER NOT NULL,
        FOREIGN KEY(LayerId) REFERENCES Layers(Id) ON DELETE CASCADE,
        UNIQUE(LayerId, ParameterIndex)
    )");
```

**Tensors** — веса и смещения:

- `LayerId` — связь со слоем
- `ParameterIndex` — порядковый номер параметра (0 — веса, 1 — смещения)
- `ShapeJson` — размерность тензора в JSON (удобно хранить и читать)
- `DataBlob` — бинарные данные тензора (массив double)
- `DataLength` — длина массива (для проверки)

```csharp
// Индексы для быстрого поиска
_connection.ExecuteNonQuery(
    "CREATE INDEX IF NOT EXISTS idx_models_identifier ON Models(Identifier)");
_connection.ExecuteNonQuery(
    "CREATE INDEX IF NOT EXISTS idx_layers_modelid ON Layers(ModelId)");
_connection.ExecuteNonQuery(
    "CREATE INDEX IF NOT EXISTS idx_tensors_layerid ON Tensors(LayerId)");
```

Индексы ускоряют поиск по идентификатору и связям между таблицами.

## Save: Сохраняем модель в БД

```csharp
public void Save(Model model, string identifier)
{
    _connection.Open();
    _connection.BeginTransaction();

    try
    {
        // Удаляем старую версию если есть
        if (Exists(identifier))
        {
            DeleteExistingModel(identifier);
        }

        // Сохраняем метаданные модели
        int modelId = SaveModelMetadata(identifier, layers.Count, totalParameters);

        // Сохраняем слои и их тензоры
        for (int i = 0; i < layers.Count; i++)
        {
            var layer = layers[i];
            int layerId = SaveLayer(modelId, layer, i);

            // Сохраняем тензоры (веса, смещения)
            for (int j = 0; j < layer.Parameters.Count; j++)
            {
                SaveTensor(layerId, layer.Parameters[j], j);
            }
        }

        _connection.CommitTransaction();
    }
    catch
    {
        _connection.RollbackTransaction();
        throw;
    }
    finally
    {
        _connection.Close();
    }
}
```

**Важные моменты:**

- Используем транзакцию, чтобы все операции выполнились атомарно
- Если что-то пошло не так — откатываем изменения
- При сохранении новой версии удаляем старую (можно изменить на версионирование)

### Сохранение метаданных модели

```csharp
private int SaveModelMetadata(string identifier, int totalLayers, int totalParameters)
{
    _connection.ExecuteNonQuery(@"
        INSERT INTO Models (Identifier, CreatedAt, Version, TotalLayers, TotalParameters)
        VALUES (@id, @createdAt, @version, @totalLayers, @totalParameters)",
        ("id", identifier),
        ("createdAt", DateTime.UtcNow),
        ("version", 1),
        ("totalLayers", totalLayers),
        ("totalParameters", totalParameters));

    return (int)_connection.ExecuteScalar<long>("SELECT last_insert_rowid()");
}
```

После вставки получаем ID новой записи через `last_insert_rowid()` — это стандартный способ в SQLite.

### Сохранение слоя

```csharp
private int SaveLayer(int modelId, Layer layer, int index)
{
    string sql = @"
        INSERT INTO Layers (
            ModelId, LayerIndex, LayerType,
            InputSize, OutputSize,
            InputChannels, OutputChannels, KernelSize, Stride, Padding
        ) VALUES (
            @modelId, @index, @type,
            @inputSize, @outputSize,
            @inputChannels, @outputChannels, @kernelSize, @stride, @padding
        )";

    var parameters = new List<(string, object)>
    {
        ("modelId", modelId),
        ("index", index),
        ("type", layer.GetType().Name)
    };

    // Добавляем специфичные параметры
    switch (layer)
    {
        case LinearLayer linear:
            parameters.Add(("inputSize", linear.InputSize));
            parameters.Add(("outputSize", linear.OutputSize));
            // остальные NULL
            break;
```

Для разных типов слоёв заполняем соответствующие поля. Остальные остаются NULL.

### Сохранение тензора

```csharp
private void SaveTensor(int layerId, Tensor tensor, int parameterIndex)
{
    string shapeJson = JsonSerializer.Serialize(tensor.Shape, _jsonOptions);
    byte[] dataBlob = SerializeDoubles(tensor.Data);

    _connection.ExecuteNonQuery(@"
        INSERT INTO Tensors (LayerId, ParameterIndex, ShapeJson, DataBlob, DataLength)
        VALUES (@layerId, @paramIndex, @shapeJson, @dataBlob, @dataLength)",
        ("layerId", layerId),
        ("paramIndex", parameterIndex),
        ("shapeJson", shapeJson),
        ("dataBlob", dataBlob),
        ("dataLength", tensor.Size));
}
```

Размерность храним в JSON (человекочитаемо), данные — в бинарном виде (компактно и быстро).

## Load: Загружаем модель из БД

```csharp
public Model Load(string identifier)
{
    _connection.Open();

    try
    {
        // Загружаем модель
        var modelData = LoadModelMetadata(identifier);
        if (modelData == null)
            throw new InvalidOperationException($"Модель {identifier} не найдена");

        var model = new Model();

        // Загружаем слои по порядку
        var layers = LoadLayers(modelData.Id);
        foreach (var layerData in layers.OrderBy(l => l.LayerIndex))
        {
            var layer = CreateLayerFromDb(layerData);

            // Загружаем тензоры для слоя
            var tensors = LoadTensors(layerData.Id);
            foreach (var tensorData in tensors.OrderBy(t => t.ParameterIndex))
            {
                var tensor = layer.Parameters[tensorData.ParameterIndex];
                LoadTensorData(tensor, tensorData);
            }

            model.AddLayer(layer);
        }

        return model;
    }
    finally
    {
        _connection.Close();
    }
}
```

Процесс зеркален сохранению:

1. Загружаем метаданные модели
2. Загружаем все слои этой модели
3. Для каждого слоя загружаем его тензоры
4. Восстанавливаем слой и тензоры

### Загрузка метаданных

```csharp
private DbModel? LoadModelMetadata(string identifier)
{
    return _connection.ExecuteQuery(@"
        SELECT Id, Identifier, CreatedAt, Description, Version, TotalLayers, TotalParameters
        FROM Models WHERE Identifier = @id",
        reader => new DbModel
        {
            Id = reader.GetInt32(0),
            Identifier = reader.GetString(1),
            CreatedAt = reader.GetDateTime(2),
            Description = reader.IsDBNull(3) ? null : reader.GetString(3),
            Version = reader.GetInt32(4),
            TotalLayers = reader.GetInt32(5),
            TotalParameters = reader.GetInt32(6)
        },
        ("id", identifier)).FirstOrDefault();
}
```

Используем маппер для преобразования IDataReader в объект.

### Загрузка слоёв

```csharp
private List<DbLayer> LoadLayers(int modelId)
{
    return _connection.ExecuteQuery(@"
        SELECT Id, LayerIndex, LayerType,
               InputSize, OutputSize,
               InputChannels, OutputChannels, KernelSize, Stride, Padding
        FROM Layers WHERE ModelId = @modelId
        ORDER BY LayerIndex",
        reader => new DbLayer
        {
            Id = reader.GetInt32(0),
            LayerIndex = reader.GetInt32(1),
            LayerType = reader.GetString(2),
            InputSize = reader.IsDBNull(3) ? null : reader.GetInt32(3),
            OutputSize = reader.IsDBNull(4) ? null : reader.GetInt32(4),
            InputChannels = reader.IsDBNull(5) ? null : reader.GetInt32(5),
            OutputChannels = reader.IsDBNull(6) ? null : reader.GetInt32(6),
            KernelSize = reader.IsDBNull(7) ? null : reader.GetInt32(7),
            Stride = reader.IsDBNull(8) ? null : reader.GetInt32(8),
            Padding = reader.IsDBNull(9) ? null : reader.GetInt32(9)
        },
        ("modelId", modelId)).ToList();
}
```

### Загрузка тензоров

```csharp
private void LoadTensorData(Tensor tensor, DbTensor tensorData)
{
    // Проверяем форму
    var shape = JsonSerializer.Deserialize<int[]>(tensorData.ShapeJson, _jsonOptions);
    if (shape == null || !tensor.Shape.SequenceEqual(shape))
        throw new InvalidOperationException("Несовпадение размерностей");

    // Загружаем данные
    var data = DeserializeDoubles(tensorData.DataBlob, tensorData.DataLength);
    Array.Copy(data, tensor.Data, data.Length);
}
```

Сверяем форму и копируем данные в тензор.

## Сериализация double[] в byte[]

```csharp
private byte[] SerializeDoubles(double[] data)
{
    byte[] bytes = new byte[data.Length * sizeof(double)];
    Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
    return bytes;
}

private double[] DeserializeDoubles(byte[] bytes, int expectedLength)
{
    double[] data = new double[expectedLength];
    Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
    return data;
}
```

`Buffer.BlockCopy` — самый быстрый способ скопировать массив значений в массив байтов и обратно. Работает на уровне памяти, без циклов.

## Преимущества базы данных

1. **Структурированное хранение** — всё разложено по таблицам
2. **Быстрый поиск** — по идентификатору, дате, количеству слоёв
3. **Версионирование** — можно хранить несколько версий одной модели
4. **Транзакции** — гарантия целостности данных
5. **Масштабирование** — от SQLite на локальной машине до PostgreSQL на сервере
6. **Метаданные** — можно добавлять любые поля (описание, гиперпараметры, метрики)

## Недостатки

1. **Сложность** — больше кода, больше движущихся частей
2. **Зависимость от СУБД** — нужно устанавливать и настраивать
3. **Производительность** — медленнее прямого файлового доступа
4. **Порог входа** — нужно понимать SQL

## Пример использования

```csharp
// Создаем подключение к SQLite
var connection = new SqliteConnection("./models.db");

// Создаем сохранятор
var dbSaver = new DatabaseModelSaver(connection);

// Сохраняем модель
dbSaver.Save(model, "mnist_cnn_v1");

// Проверяем существование
if (dbSaver.Exists("mnist_cnn_v1"))
{
    Console.WriteLine("Модель есть в базе!");
}

// Загружаем модель
var loadedModel = dbSaver.Load("mnist_cnn_v1");
```

## Сравнение всех сохраняторов

| Характеристика      | Binary     | JSON       | Database   |
| ------------------- | ---------- | ---------- | ---------- |
| **Скорость**        | ⭐⭐⭐⭐⭐ | ⭐⭐       | ⭐⭐⭐     |
| **Размер**          | ⭐⭐⭐⭐⭐ | ⭐⭐       | ⭐⭐⭐⭐   |
| **Читаемость**      | ⭐         | ⭐⭐⭐⭐⭐ | ⭐⭐       |
| **Поиск**           | ⭐         | ⭐         | ⭐⭐⭐⭐⭐ |
| **Метаданные**      | ⭐         | ⭐⭐       | ⭐⭐⭐⭐⭐ |
| **Сложность**       | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐   | ⭐⭐       |
| **Версионирование** | ⭐         | ⭐⭐       | ⭐⭐⭐⭐⭐ |

## Заключение: База данных как центральное хранилище

DatabaseModelSaver открывает новые возможности:

- Храните тысячи моделей в одном месте
- Ищите по любым критериям
- Добавляйте метаданные (автор, дата, гиперпараметры, метрики)
- Делитесь доступом с коллегами
- Интегрируйте с веб-сервисами

Это уже не просто сохранятор, а **система управления моделями** (Model Registry). В следующих статьях мы добавим версионирование, теги, поиск и веб-интерфейс.

А пока у нас есть три инструмента для разных задач:

- **BinaryModelSaver** — для быстрого сохранения чекпоинтов
- **JsonModelSaver** — для отладки и обмена
- **DatabaseModelSaver** — для серьезного управления моделями

И все через единый интерфейс IModelSaver. Подменил реализацию — и вся система работает по-новому. Вот что значит правильная архитектура!
