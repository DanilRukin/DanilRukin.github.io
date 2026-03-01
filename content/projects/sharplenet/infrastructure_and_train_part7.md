---
title: "SharpLeNet. BinaryModelSaver: Когда важны скорость и компактность"
date: 2026-03-01
description: "SharpLeNet. BinaryModelSaver: Когда важны скорость и компактность"
tags: ["SharpLeNet"]
categories: ["SharpLeNet"]
series: ["SharpLeNet"]
reading_time: 5
featured: true
---

Друзья, мы определили интерфейс для сохранения моделей и создали базовый callback для чекпоинтов. Теперь пришло время реализовать первого конкретного сохранятора — **бинарного**.

Почему бинарный формат? Представьте, что вам нужно сохранить модель с миллионами параметров. В текстовом формате (JSON, XML) это будет огромный файл, который медленно читается и пишется. Бинарный формат — это как черный ящик: человеку не прочитать, но компьютер работает с ним максимально быстро и эффективно.

## Конструктор: Куда будем сохранять?

```csharp
public class BinaryModelSaver : IModelSaver
{
    private readonly string _directory;

    public BinaryModelSaver(string directory)
    {
        _directory = directory;
        Directory.CreateDirectory(directory);
    }
```

Здесь всё просто: передаем директорию, куда будем сохранять модели. Если директории нет — создаем. Дальше все файлы будут ложиться в эту папку.

## Save: Превращаем модель в байты

```csharp
public void Save(Model model, string identifier)
{
    string filePath = Path.Combine(_directory, $"{identifier}.bin");

    using var stream = File.Create(filePath);
    using var writer = new BinaryWriter(stream);

    // Получаем слои модели через рефлексию
    var layers = GetModelLayers(model);

    // Сохраняем количество слоев
    writer.Write(layers.Count);

    foreach (var layer in layers)
    {
        // Сохраняем тип слоя
        writer.Write(layer.GetType().FullName ?? "");

        // Сохраняем параметры слоя (веса, смещения)
        SaveLayer(layer, writer);

        // Сохраняем веса
        foreach (var param in layer.Parameters)
        {
            SaveTensor(param, writer);
        }
    }
}
```

**Разберем по шагам:**

1. **Создаем файл** с именем `{identifier}.bin` в указанной директории
2. **Открываем BinaryWriter** — это стандартный инструмент .NET для записи примитивных типов в бинарном виде
3. **Получаем список слоев** через рефлексию (да, пришлось залезть в приватное поле)
4. **Сохраняем количество слоев** — при загрузке будем знать, сколько читать
5. **Для каждого слоя:**
   - Сохраняем полное имя типа (чтобы потом создать правильный объект)
   - Сохраняем параметры слоя (размеры, каналы и т.д.)
   - Сохраняем все тензоры параметров (веса и смещения)

## GetModelLayers: Рефлексия — не зло, а необходимость

```csharp
private List<Layer> GetModelLayers(Model model)
{
    var field = typeof(Model).GetField("_layers",
        System.Reflection.BindingFlags.NonPublic |
        System.Reflection.BindingFlags.Instance);

    return field?.GetValue(model) as List<Layer> ?? new List<Layer>();
}
```

Да, мы лезем в приватное поле. Обычно это не рекомендуется, но здесь у нас нет выбора — список слоев не доступен публично. В продакшене стоило бы добавить публичное свойство `Layers` в класс `Model`, но для учебного примера — сойдёт.

## SaveLayer: Архитектура слоя в бинарном виде

```csharp
private void SaveLayer(Layer layer, BinaryWriter writer)
{
    switch (layer)
    {
        case LinearLayer linear:
            writer.Write(linear.InputSize);
            writer.Write(linear.OutputSize);
            break;

        case Conv2DLayer conv:
            writer.Write(conv.InputChannels);
            writer.Write(conv.OutputChannels);
            writer.Write(conv.KernelSize);
            writer.Write(conv.Stride);
            writer.Write(conv.Padding);
            break;

        // Для слоев без параметров ничего не сохраняем
        case ReLULayer:
        case SoftmaxLayer:
        case FlattenLayer:
        case MaxPoolingLayer:
        case AvgPoolingLayer:
            break;

        default:
            throw new Exception($"Неизвестный тип слоя: {layer.GetType()}");
    }
}
```

У каждого типа слоя свои параметры конструкции:

- **LinearLayer** — нужно знать входной и выходной размер
- **Conv2DLayer** — каналы, размер ядра, шаг, паддинг
- **Активации и пулинг** — не имеют параметров (кроме пулинга, но у нас они жестко заданы)

Эти параметры нужны, чтобы при загрузке воссоздать слой с правильной архитектурой.

## Load: Восстанавливаем модель из байтов

```csharp
public Model Load(string identifier)
{
    string filePath = Path.Combine(_directory, $"{identifier}.bin");
    if (!File.Exists(filePath))
        throw new FileNotFoundException($"Модель {identifier} не найдена");

    using var stream = File.OpenRead(filePath);
    using var reader = new BinaryReader(stream);

    int layerCount = reader.ReadInt32();
    var model = new Model();

    for (int i = 0; i < layerCount; i++)
    {
        string layerTypeName = reader.ReadString();
        var layer = CreateLayer(layerTypeName, reader);

        // Загружаем веса
        foreach (var param in layer.Parameters)
        {
            LoadTensor(param, reader);
        }

        model.AddLayer(layer);
    }

    return model;
}
```

Процесс загрузки — зеркальное отражение сохранения:

1. Читаем количество слоев
2. Для каждого слоя:
   - Читаем имя типа
   - Создаем слой с правильными параметрами (через `CreateLayer`)
   - Загружаем веса в тензоры слоя
   - Добавляем слой в модель

## CreateLayer: Фабрика слоев

```csharp
private Layer CreateLayer(string typeName, BinaryReader reader)
{
    return typeName switch
    {
        nameof(LinearLayer) => new LinearLayer(
            reader.ReadInt32(), // inputSize
            reader.ReadInt32()  // outputSize
        ),

        nameof(Conv2DLayer) => new Conv2DLayer(
            reader.ReadInt32(), // inputChannels
            reader.ReadInt32(), // outputChannels
            reader.ReadInt32(), // kernelSize
            reader.ReadInt32(), // stride
            reader.ReadInt32()  // padding
        ),

        nameof(ReLULayer) => new ReLULayer(),
        nameof(SoftmaxLayer) => new SoftmaxLayer(),
        nameof(FlattenLayer) => new FlattenLayer(),
        nameof(MaxPoolingLayer) => new MaxPoolingLayer(),
        nameof(AvgPoolingLayer) => new AvgPoolingLayer(),

        _ => throw new Exception($"Неизвестный тип слоя: {typeName}")
    };
}
```

Здесь мы по имени типа создаем соответствующий слой. Для слоев с параметрами читаем их из потока в том же порядке, в котором сохраняли. Для слоев без параметров просто вызываем конструктор по умолчанию.

## SaveTensor и LoadTensor: Сериализация тензоров

```csharp
private void SaveTensor(Tensor tensor, BinaryWriter writer)
{
    // Сохраняем размерность
    writer.Write(tensor.Rank);
    foreach (int dim in tensor.Shape)
        writer.Write(dim);

    // Сохраняем данные
    foreach (double value in tensor.Data)
        writer.Write(value);
}

private void LoadTensor(Tensor tensor, BinaryReader reader)
{
    // Проверяем размерность
    int rank = reader.ReadInt32();
    var shape = new int[rank];
    for (int i = 0; i < rank; i++)
        shape[i] = reader.ReadInt32();

    if (!tensor.Shape.SequenceEqual(shape))
        throw new Exception("Несовпадение формы тензора");

    // Загружаем данные
    for (int i = 0; i < tensor.Size; i++)
        tensor.Data[i] = reader.ReadDouble();
}
```

**Важный момент:** При загрузке мы проверяем, что сохраненная форма совпадает с формой тензора в созданном слое. Если нет — кидаем исключение. Это защита от битых файлов или несовместимых версий.

## Exists: Проверка наличия

```csharp
public bool Exists(string identifier)
{
    return File.Exists(Path.Combine(_directory, $"{identifier}.bin"));
}
```

Тривиальная проверка — существует ли файл с таким идентификатором.

## Пример использования

```csharp
// Создаем сохранятор
var saver = new BinaryModelSaver("./models");

// Обучаем модель и сохраняем лучшую версию
var checkpoint = new ModelCheckpoint(model, saver, "mnist_cnn");
trainer.AddCallback(checkpoint);
trainer.Fit(10);

// Где-то в другом месте загружаем сохраненную модель
var loadedModel = saver.Load("epoch_8_loss_0.123456");

// Используем для предсказаний
var predictions = loadedModel.Forward(testData);
```

## Преимущества бинарного формата

1. **Скорость** — чтение и запись примитивных типов происходит максимально быстро
2. **Компактность** — никаких лишних символов, только голые данные
3. **Точность** — double сохраняется как double, без потерь при конвертации в текст
4. **Простота** — минимум кода, легко отлаживать

## Недостатки

1. **Нечитаемость** — нельзя открыть в блокноте и посмотреть веса
2. **Несовместимость версий** — если изменится формат, старые файлы не прочитаются
3. **Платформозависимость** — теоретически могут быть проблемы при переносе между разными архитектурами (но .NET старается их избегать)

## Что дальше?

В следующих статьях мы реализуем:

- **JsonModelSaver** — человекочитаемый формат для отладки
- **DatabaseModelSaver** — для продакшн-систем
- Версионирование моделей
- Сохранение метаданных (история обучения, гиперпараметры)

## Заключение: Модель обретает плоть

BinaryModelSaver — это наш первый, самый простой, но очень эффективный способ дать моделям бессмертие. Он не пытается быть красивым или человекочитаемым. Его задача — быстро и компактно сохранить веса, чтобы потом так же быстро их восстановить.

В паре с ModelCheckpoint мы получаем мощный механизм: во время обучения автоматически сохраняется лучшая версия модели, а потом её можно загрузить и использовать где угодно. Никакого повторного обучения, никакой потери прогресса.

И всё это через единый интерфейс IModelSaver. Хотите другой формат? Просто реализуйте интерфейс по-новому, а весь остальной код останется без изменений. Вот что значит правильная архитектура!
