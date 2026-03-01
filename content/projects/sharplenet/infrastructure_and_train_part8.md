---
title: "SharpLeNet. JsonModelSaver: Когда хочется заглянуть под капот"
date: 2026-03-01
description: "SharpLeNet. JsonModelSaver: Когда хочется заглянуть под капот"
tags: ["SharpLeNet"]
categories: ["SharpLeNet"]
series: ["SharpLeNet"]
reading_time: 5
featured: true
---

Друзья, мы уже умеем сохранять модели в бинарном формате — быстро, компактно, но совершенно непонятно для человека. Открыть такой файл в блокноте — всё равно что читать книгу на незнакомом языке. Иероглифы, кракозябры и никакого смысла.

Но бывают ситуации, когда хочется **увидеть**, что там внутри. Например:

- Вы отлаживаете модель и хотите проверить значения весов
- Вы хотите поделиться моделью с коллегой, чтобы он мог её прочитать
- Вы сохраняете чекпоинты для экспериментов и хотите быстро сравнивать их
- Вы подозреваете, что веса испортились, и хотите это проверить

Для таких случаев нужен человекочитаемый формат. И король таких форматов — **JSON**.

## JsonModelSaver: Архитектура

```csharp
public class JsonModelSaver : IModelSaver
{
    private readonly string _directory;
    private readonly JsonSerializerOptions _jsonOptions;

    public JsonModelSaver(string directory)
    {
        _directory = directory;
        Directory.CreateDirectory(directory);

        _jsonOptions = new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        };
    }
```

**Что здесь важного:**

- `WriteIndented = true` — делаем JSON красивым, с отступами и переносами строк
- `PropertyNamingPolicy = CamelCase` — имена свойств в стиле camelCase (как принято в JSON)
- `DefaultIgnoreCondition` — не пишем null-поля, экономим место

## Save: Превращаем модель в красивый текст

```csharp
public void Save(Model model, string identifier)
{
    if (model == null) throw new ArgumentNullException(nameof(model));
    if (string.IsNullOrWhiteSpace(identifier))
        throw new ArgumentException("Идентификатор модели не может быть пустым", nameof(identifier));

    string filePath = Path.Combine(_directory, $"{identifier}.json");

    var modelData = new ModelData();
    var layers = GetModelLayers(model);

    foreach (var layer in layers)
    {
        var layerData = new LayerData
        {
            Type = layer.GetType().Name
        };

        // Сохраняем веса и смещения
        foreach (var param in layer.Parameters)
        {
            layerData.Parameters.Add(new TensorData
            {
                Shape = param.Shape.ToArray(),
                Data = param.Data.ToArray()
            });
        }
```

Первый шаг — создаем объект `ModelData`, который будет сериализоваться в JSON. Для каждого слоя:

1. Запоминаем тип слоя (`LinearLayer`, `Conv2DLayer` и т.д.)
2. Сохраняем все параметры (веса и смещения) в виде массива чисел и их размерностей

### Сохранение специфичных параметров

```csharp
// Сохраняем специфичные параметры слоя
switch (layer)
{
    case LinearLayer linear:
        layerData.LinearParams = new LinearParams
        {
            InputSize = linear.InputSize,
            OutputSize = linear.OutputSize
        };
        break;

    case Conv2DLayer conv:
        layerData.ConvParams = new ConvParams
        {
            InputChannels = conv.InputChannels,
            OutputChannels = conv.OutputChannels,
            KernelSize = conv.KernelSize,
            Stride = conv.Stride,
            Padding = conv.Padding
        };
        break;
```

Разные слои требуют разных параметров для создания. Линейному слою нужно знать входной и выходной размер. Сверточному — каналы, размер ядра, шаг и паддинг. Мы сохраняем эти параметры в специальных DTO-объектах.

### Добавление метаданных

```csharp
// Добавляем метаданные
modelData.Metadata["modelType"] = model.GetType().Name;
modelData.Metadata["totalLayers"] = layers.Count.ToString();
modelData.Metadata["totalParameters"] = model.Parameters.Sum(p => p.Size).ToString();

string json = JsonSerializer.Serialize(modelData, _jsonOptions);
File.WriteAllText(filePath, json);
```

В конце добавляем метаданные — полезную информацию о модели, которая может пригодиться без загрузки всей модели.

## Что получается на выходе?

```json
{
  "metadata": {
    "modelType": "Sequential",
    "totalLayers": "3",
    "totalParameters": "101770"
  },
  "layers": [
    {
      "type": "LinearLayer",
      "linearParams": {
        "inputSize": 784,
        "outputSize": 128
      },
      "parameters": [
        {
          "shape": [128, 784],
          "data": [0.123, -0.456, 0.789, ...]
        },
        {
          "shape": [128],
          "data": [0.001, -0.002, ...]
        }
      ]
    },
    {
      "type": "ReLULayer",
      "parameters": []
    },
    {
      "type": "LinearLayer",
      "linearParams": {
        "inputSize": 128,
        "outputSize": 10
      },
      "parameters": [
        {
          "shape": [10, 128],
          "data": [0.234, -0.567, ...]
        },
        {
          "shape": [10],
          "data": [0.001, -0.002, ...]
        }
      ]
    }
  ]
}
```

Красота! Можно открыть в любом текстовом редакторе и увидеть:

- Архитектуру модели (какие слои идут подряд)
- Размерности всех тензоров
- Значения весов (хотя их много, но можно пролистать)
- Метаданные

## Load: Собираем модель обратно

```csharp
public Model Load(string identifier)
{
    if (string.IsNullOrWhiteSpace(identifier))
        throw new ArgumentException("Идентификатор модели не может быть пустым", nameof(identifier));

    string filePath = Path.Combine(_directory, $"{identifier}.json");
    if (!File.Exists(filePath))
        throw new FileNotFoundException($"Модель {identifier} не найдена в {filePath}");

    string json = File.ReadAllText(filePath);
    var modelData = JsonSerializer.Deserialize<ModelData>(json, _jsonOptions);

    if (modelData == null)
        throw new InvalidOperationException("Не удалось десериализовать модель");

    var model = new Model();

    foreach (var layerData in modelData.Layers)
    {
        Layer layer = CreateLayerFromData(layerData);

        // Загружаем веса
        for (int i = 0; i < layer.Parameters.Count; i++)
        {
            if (i >= layerData.Parameters.Count)
                throw new InvalidOperationException($"Недостаточное кол-во параметров для слоя {layerData.Type}");

            var tensorData = layerData.Parameters[i];

            // Проверяем форму
            if (!layer.Parameters[i].Shape.SequenceEqual(tensorData.Shape))
                throw new InvalidOperationException(
                    $"Несовпадение размерностей слоя {layerData.Type} параметр {i}. " +
                    $"Ожидалось [{string.Join(", ", layer.Parameters[i].Shape)}], " +
                    $"получено [{string.Join(", ", tensorData.Shape)}]");

            // Проверяем размер данных
            if (layer.Parameters[i].Size != tensorData.Data.Length)
                throw new InvalidOperationException(
                    $"Несоответствие размера данных для слоя {layerData.Type} параметр {i}. " +
                    $"Ожидалось {layer.Parameters[i].Size}, получено {tensorData.Data.Length}");

            Array.Copy(tensorData.Data, layer.Parameters[i].Data, tensorData.Data.Length);
        }

        model.AddLayer(layer);
    }

    return model;
}
```

**Важные проверки при загрузке:**

1. **Проверка размерностей** — убеждаемся, что сохраненная форма тензора совпадает с формой в созданном слое
2. **Проверка размера данных** — количество чисел должно совпадать с ожидаемым
3. **Проверка количества параметров** — у слоя должно быть столько же параметров, сколько сохранено

Эти проверки защищают от битых файлов и несовместимых версий.

## CreateLayerFromData: Фабрика слоёв

```csharp
private Layer CreateLayerFromData(LayerData layerData)
{
    return layerData.Type switch
    {
        nameof(LinearLayer) => CreateLinearLayer(layerData),
        nameof(Conv2DLayer) => CreateConvLayer(layerData),
        nameof(ReLULayer) => new ReLULayer(),
        nameof(SoftmaxLayer) => new SoftmaxLayer(),
        nameof(FlattenLayer) => new FlattenLayer(),
        nameof(MaxPoolingLayer) => new MaxPoolingLayer(),
        nameof(AvgPoolingLayer) => new AvgPoolingLayer(),
        _ => throw new NotSupportedException($"Неподдерживаемый слой: {layerData.Type}")
    };
}

private LinearLayer CreateLinearLayer(LayerData layerData)
{
    if (layerData.LinearParams == null)
        throw new InvalidOperationException("Пропущены параметры для линейного слоя");

    var layer = new LinearLayer(
        layerData.LinearParams.InputSize,
        layerData.LinearParams.OutputSize
    );

    return layer;
}

private Conv2DLayer CreateConvLayer(LayerData layerData)
{
    if (layerData.ConvParams == null)
        throw new InvalidOperationException("Пропущены параметры сверточного слоя");

    var layer = new Conv2DLayer(
        layerData.ConvParams.InputChannels,
        layerData.ConvParams.OutputChannels,
        layerData.ConvParams.KernelSize,
        layerData.ConvParams.Stride,
        layerData.ConvParams.Padding
    );

    return layer;
}
```

Здесь мы воссоздаем слои с правильными параметрами, прежде чем загружать в них веса.

## Полезные методы для работы с моделями

### Получение списка сохраненных моделей

```csharp
public string[] GetSavedModels()
{
    return Directory.GetFiles(_directory, "*.json")
        .Select(f => Path.GetFileNameWithoutExtension(f))
        .ToArray();
}
```

Удобно, когда нужно показать пользователю, какие модели доступны для загрузки.

### Удаление модели

```csharp
public void Delete(string identifier)
{
    string filePath = Path.Combine(_directory, $"{identifier}.json");
    if (File.Exists(filePath))
        File.Delete(filePath);
}
```

Чтобы не засорять диск старыми чекпоинтами.

### Получение метаданных без загрузки всей модели

```csharp
public ModelMetadata GetMetadata(string identifier)
{
    string filePath = Path.Combine(_directory, $"{identifier}.json");
    if (!File.Exists(filePath))
        return null;

    string json = File.ReadAllText(filePath);
    var modelData = JsonSerializer.Deserialize<ModelData>(json, _jsonOptions);

    return new ModelMetadata
    {
        Identifier = identifier,
        CreatedAt = modelData?.CreatedAt ?? DateTime.MinValue,
        TotalLayers = modelData?.Layers.Count ?? 0,
        Metadata = modelData?.Metadata ?? new Dictionary<string, string>()
    };
}
```

Очень полезно! Можно быстро узнать, сколько слоёв в модели, когда она создана, не загружая все веса в память.

## Сравнение с бинарным форматом

| Характеристика             | BinaryModelSaver | JsonModelSaver      |
| -------------------------- | ---------------- | ------------------- |
| **Размер файла**           | Маленький        | Большой (в 2-5 раз) |
| **Скорость сохранения**    | Очень быстрая    | Медленнее           |
| **Скорость загрузки**      | Очень быстрая    | Медленнее           |
| **Читаемость**             | Никакой          | Отличная            |
| **Отладка**                | Сложная          | Легкая              |
| **Совместимость версий**   | Хрупкая          | Гибкая              |
| **Редактирование вручную** | Нельзя           | Можно               |

## Пример использования

```csharp
// Создаем JSON-сохранятор
var jsonSaver = new JsonModelSaver("./models_json");

// Сохраняем модель
jsonSaver.Save(model, "mnist_cnn_v1");

// Смотрим список моделей
var models = jsonSaver.GetSavedModels();
foreach (var modelName in models)
{
    var meta = jsonSaver.GetMetadata(modelName);
    Console.WriteLine($"{modelName}: {meta.TotalLayers} слоёв, создана {meta.CreatedAt}");
}

// Загружаем модель
var loadedModel = jsonSaver.Load("mnist_cnn_v1");

// Если надоела - удаляем
jsonSaver.Delete("mnist_cnn_v1");
```

## Преимущества JSON-формата

1. **Человекочитаемость** — можно открыть и посмотреть, что внутри
2. **Отладка** — легко проверить, правильно ли сохранились веса
3. **Версионирование** — можно хранить в Git (для маленьких моделей)
4. **Совместимость** — JSON понимают все языки программирования
5. **Редактирование** — можно вручную поправить веса (для экспериментов)

## Недостатки

1. **Размер** — гораздо больше бинарного
2. **Скорость** — сериализация/десериализация медленнее
3. **Потеря точности** — при записи чисел в текст теряется точность (но для double это не критично)
4. **Не подходит для больших моделей** — миллионы весов в тексте будут весить сотни мегабайт

## Когда использовать JSON?

- **Для отладки** — когда нужно убедиться, что веса сохраняются правильно
- **Для маленьких моделей** — до нескольких тысяч параметров
- **Для обмена с коллегами** — чтобы они могли открыть и посмотреть
- **Для экспериментов** — когда нужно вручную подправить веса
- **Для документации** — показать архитектуру модели в читаемом виде

## А когда бинарный формат?

- **Для больших моделей** — миллионы параметров
- **Для продакшна** — где важны скорость и размер
- **Для чекпоинтов** — во время долгого обучения
- **Для распространения** — когда не нужно, чтобы пользователи копались внутри

## Заключение: Два сапога — пара

BinaryModelSaver и JsonModelSaver — это не конкуренты, а два инструмента для разных задач. Как молоток и отвертка: каждый для своего.

- **BinaryModelSaver** — для работы, для продакшна, для скорости
- **JsonModelSaver** — для отладки, для экспериментов, для общения

Благодаря единому интерфейсу `IModelSaver` мы можем легко переключаться между ними, не меняя остальной код. Хотите быстро сохранить чекпоинт во время обучения — берите бинарный. Хотите потом посмотреть, что там внутри, — переключитесь на JSON.
