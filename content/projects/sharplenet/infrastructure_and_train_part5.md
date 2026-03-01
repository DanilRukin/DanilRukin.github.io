---
title: "SharpLeNet. Датасеты и Даталоадеры: Разделение ответственности (исправленная версия)"
date: 2026-03-01
description: "SharpLeNet. Датасеты и Даталоадеры: Разделение ответственности (исправленная версия)"
tags: ["SharpLeNet"]
categories: ["SharpLeNet"]
series: ["SharpLeNet"]
reading_time: 5
featured: true
---

Друзья, в программировании, как и в жизни, иногда приходится возвращаться и переосмысливать свои решения. Это нормально! Это называется рефакторингом, и это признак роста, а не ошибки.

В прошлой статье я смешал две разные концепции в одном классе. Сегодня мы исправим эту ошибку и построим правильную архитектуру, где **Dataset** отвечает за доступ к данным, а **DataLoader** — за то, как эти данные подаются в модель.

## Dataset vs DataLoader

Давайте проведем четкую границу:

| Компонент      | Ответственность                                            | Аналогия                                    |
| -------------- | ---------------------------------------------------------- | ------------------------------------------- |
| **Dataset**    | "Что у нас есть?" — хранение и доступ к отдельным примерам | Библиотека с книгами                        |
| **DataLoader** | "Как мы это подаем?" — батчи, перемешивание, порядок       | Библиотекарь, который выдает книги порциями |

**Dataset** знать не знает о батчах, shuffle и прочих деталях обучения. Его задача — по индексу вернуть один конкретный пример. Всё.

**DataLoader** берет Dataset и добавляет логику итерации: разбивку на батчи, перемешивание, пропуск последнего неполного батча.

## Dataset: Абстрактный доступ к данным

```csharp
public abstract class Dataset : IDisposable
{
    /// <summary>
    /// Количество примеров в датасете
    /// </summary>
    public abstract int Count { get; }

    /// <summary>
    /// Возвращает один пример по индексу
    /// </summary>
    public abstract (Tensor features, Tensor labels) GetItem(int index);

    /// <summary>
    /// Форма признаков для одного примера
    /// </summary>
    public abstract int[] FeatureShape { get; }

    /// <summary>
    /// Форма меток для одного примера
    /// </summary>
    public abstract int[] LabelShape { get; }

    public virtual void Dispose() { }
}
```

**Ключевые моменты:**

- `Count` — сколько всего примеров
- `GetItem(index)` — получить один пример по индексу (это единственный метод доступа к данным!)
- `FeatureShape` и `LabelShape` — нужны, чтобы DataLoader знал, какой формы создавать батчи

Никаких батчей, никакого shuffle, никаких итераторов. Только базовый доступ.

## Вариант 1: InMemoryMNISTDataset — всё в памяти

```csharp
public class InMemoryMNISTDataset : Dataset
{
    private readonly Tensor _features;
    private readonly Tensor _labels;
    private readonly int[] _featureShape;
    private readonly int[] _labelShape;

    public override int Count { get; }
    public override int[] FeatureShape => _featureShape;
    public override int[] LabelShape => _labelShape;

    public InMemoryMNISTDataset(string imagesPath, string labelsPath, bool normalize = true)
    {
        // Читаем заголовки файлов MNIST
        using var imagesStream = File.OpenRead(imagesPath);
        using var imagesReader = new BinaryReader(imagesStream);

        int magicNumber = ReverseBytes(imagesReader.ReadInt32());
        int numSamples = ReverseBytes(imagesReader.ReadInt32());
        int rows = ReverseBytes(imagesReader.ReadInt32());
        int cols = ReverseBytes(imagesReader.ReadInt32());

        // ... проверки ...

        Count = numSamples;
        _featureShape = new int[] { 1, rows, cols }; // [C, H, W]
        _labelShape = new int[] { 10 }; // one-hot

        int imageSize = rows * cols;
        var featuresData = new double[numSamples * imageSize];
        var labelsData = new double[numSamples * 10];

        // Читаем ВСЕ данные в память
        for (int i = 0; i < numSamples; i++)
        {
            // Читаем изображение
            for (int j = 0; j < imageSize; j++)
            {
                byte pixel = imagesReader.ReadByte();
                featuresData[i * imageSize + j] = normalize ? pixel / 255.0 : pixel;
            }

            // Читаем метку
            byte label = labelsReader.ReadByte();
            labelsData[i * 10 + label] = 1.0;
        }

        _features = new Tensor(featuresData, new int[] { numSamples, 1, rows, cols });
        _labels = new Tensor(labelsData, new int[] { numSamples, 10 });
    }

    public override (Tensor features, Tensor labels) GetItem(int index)
    {
        int imageSize = _features.Size / Count;
        int labelSize = _labels.Size / Count;

        var featuresData = new double[imageSize];
        var labelsData = new double[labelSize];

        // Копируем данные для одного примера
        Array.Copy(_features.Data, index * imageSize, featuresData, 0, imageSize);
        Array.Copy(_labels.Data, index * labelSize, labelsData, 0, labelSize);

        return (
            new Tensor(featuresData, _featureShape),
            new Tensor(labelsData, _labelShape)
        );
    }
}
```

**Плюсы:** Быстрый доступ, простой код.
**Минусы:** Требует много памяти (весь датасет в RAM).

## Вариант 2: LazyMNISTDataset — ленивая загрузка

```csharp
public class LazyMNISTDataset : Dataset
{
    private readonly string _imagesPath;
    private readonly string _labelsPath;
    private readonly bool _normalize;
    private readonly int _numSamples;
    private readonly int _rows;
    private readonly int _cols;
    private readonly int[] _featureShape;
    private readonly int[] _labelShape;

    private FileStream? _imagesStream;
    private BinaryReader? _imagesReader;
    private FileStream? _labelsStream;
    private BinaryReader? _labelsReader;

    public LazyMNISTDataset(string imagesPath, string labelsPath, bool normalize = true)
    {
        _imagesPath = imagesPath;
        _labelsPath = labelsPath;
        _normalize = normalize;

        // Открываем файлы и читаем ТОЛЬКО заголовки
        _imagesStream = File.OpenRead(imagesPath);
        _imagesReader = new BinaryReader(_imagesStream);

        int magicNumber = ReverseBytes(_imagesReader.ReadInt32());
        _numSamples = ReverseBytes(_imagesReader.ReadInt32());
        _rows = ReverseBytes(_imagesReader.ReadInt32());
        _cols = ReverseBytes(_imagesReader.ReadInt32());

        // ... проверки ...

        _featureShape = new int[] { 1, _rows, _cols };
        _labelShape = new int[] { 10 };
    }

    public override (Tensor features, Tensor labels) GetItem(int index)
    {
        lock (this) // Потокобезопасность для параллельной загрузки
        {
            // Позиция изображения: заголовок (16 байт) + индекс * размер_изображения
            int imageSize = _rows * _cols;
            long imagePosition = 16 + index * imageSize;
            _imagesStream!.Seek(imagePosition, SeekOrigin.Begin);

            var imageData = new byte[imageSize];
            _imagesStream.Read(imageData, 0, imageSize);

            // Позиция метки: заголовок (8 байт) + индекс
            long labelPosition = 8 + index;
            _labelsStream!.Seek(labelPosition, SeekOrigin.Begin);
            int label = _labelsStream.ReadByte();

            // Конвертируем в double
            var featuresData = new double[imageSize];
            for (int i = 0; i < imageSize; i++)
            {
                featuresData[i] = _normalize ? imageData[i] / 255.0 : imageData[i];
            }

            var labelsData = new double[10];
            labelsData[label] = 1.0;

            return (
                new Tensor(featuresData, _featureShape),
                new Tensor(labelsData, _labelShape)
            );
        }
    }
}
```

**Плюсы:** Экономит память (данные на диске, читаются по требованию).
**Минусы:** Медленнее, требует синхронизации при многопоточности.

## DataLoader: Абстракция над итерацией

```csharp
public abstract class DataLoader : IEnumerable<(Tensor features, Tensor labels)>, IDisposable
{
    protected readonly Dataset _dataset;
    protected readonly int _batchSize;
    protected readonly bool _shuffle;

    public DataLoader(Dataset dataset, int batchSize = 32, bool shuffle = true)
    {
        _dataset = dataset ?? throw new ArgumentNullException(nameof(dataset));
        _batchSize = batchSize > 0 ? batchSize : throw new ArgumentException("Batch size must be positive");
        _shuffle = shuffle;
    }

    public abstract IEnumerator<(Tensor features, Tensor labels)> GetEnumerator();

    System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();

    public virtual void Dispose() => _dataset?.Dispose();
}
```

Обратите внимание: DataLoader ничего не знает о том, как хранятся данные. Он просто знает, что у него есть Dataset с методами `Count` и `GetItem`. Всю логику батчей и порядка он берет на себя.

## Вариант 1: DefaultDataLoader — как в PyTorch

```csharp
public class DefaultDataLoader : DataLoader
{
    private readonly Random _rnd;
    private readonly bool _dropLast;

    public DefaultDataLoader(Dataset dataset, int batchSize = 32, bool shuffle = true,
                            bool dropLast = false, int? seed = null)
        : base(dataset, batchSize, shuffle)
    {
        _dropLast = dropLast;
        _rnd = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    public override IEnumerator<(Tensor features, Tensor labels)> GetEnumerator()
    {
        // Создаем индексы для текущей эпохи
        var indices = Enumerable.Range(0, _dataset.Count).ToArray();
        if (_shuffle)
        {
            ShuffleIndices(indices);
        }

        int totalBatches = _dropLast
            ? indices.Length / _batchSize
            : (int)Math.Ceiling((double)indices.Length / _batchSize);

        for (int batchIdx = 0; batchIdx < totalBatches; batchIdx++)
        {
            int startIdx = batchIdx * _batchSize;
            int endIdx = Math.Min(startIdx + _batchSize, indices.Length);
            int currentBatchSize = endIdx - startIdx;

            // Пропускаем последний неполный батч если нужно
            if (_dropLast && currentBatchSize < _batchSize)
                break;

            yield return CreateBatch(indices, startIdx, currentBatchSize);
        }
    }

    private (Tensor features, Tensor labels) CreateBatch(int[] indices, int startIdx, int batchSize)
    {
        // Получаем первый пример для определения размеров
        var firstItem = _dataset.GetItem(0);
        int featureSize = firstItem.features.Size;
        int labelSize = firstItem.labels.Size;

        var batchFeaturesData = new double[batchSize * featureSize];
        var batchLabelsData = new double[batchSize * labelSize];

        for (int i = 0; i < batchSize; i++)
        {
            int sampleIdx = indices[startIdx + i];
            var (features, labels) = _dataset.GetItem(sampleIdx);

            Array.Copy(features.Data, 0, batchFeaturesData, i * featureSize, featureSize);
            Array.Copy(labels.Data, 0, batchLabelsData, i * labelSize, labelSize);
        }

        // Форма для батча: [batch, ...]
        var batchFeatureShape = new int[] { batchSize }.Concat(_dataset.FeatureShape).ToArray();
        var batchLabelShape = new int[] { batchSize }.Concat(_dataset.LabelShape).ToArray();

        return (
            new Tensor(batchFeaturesData, batchFeatureShape),
            new Tensor(batchLabelsData, batchLabelShape)
        );
    }

    private void ShuffleIndices(int[] indices)
    {
        for (int i = indices.Length - 1; i > 0; i--)
        {
            int j = _rnd.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }
    }
}
```

**Ключевые особенности:**

- **Shuffle** — перемешиваем индексы в начале каждой эпохи
- **DropLast** — если True, отбрасываем последний неполный батч
- **Индексы** — работаем через массив индексов, чтобы не трогать сами данные

## Вариант 2: SequentialDataLoader — без перемешивания

```csharp
public class SequentialDataLoader : DataLoader
{
    public SequentialDataLoader(Dataset dataset, int batchSize = 32)
        : base(dataset, batchSize, shuffle: false) { }

    public override IEnumerator<(Tensor features, Tensor labels)> GetEnumerator()
    {
        for (int startIdx = 0; startIdx < _dataset.Count; startIdx += _batchSize)
        {
            int endIdx = Math.Min(startIdx + _batchSize, _dataset.Count);
            int batchSize = endIdx - startIdx;

            yield return CreateBatch(startIdx, batchSize);
        }
    }

    private (Tensor features, Tensor labels) CreateBatch(int startIdx, int batchSize)
    {
        // ... аналогично DefaultDataLoader, но без индексов ...
    }
}
```

Используется для валидации или когда порядок важен (например, временные ряды).

## Как это работает вместе?

```csharp
// Создаем датасет (выбираем стратегию хранения)
Dataset trainDataset = new InMemoryMNISTDataset(
    "train-images.idx3-ubyte",
    "train-labels.idx1-ubyte"
);

// Оборачиваем в DataLoader (выбираем стратегию подачи)
DataLoader trainLoader = new DefaultDataLoader(
    trainDataset,
    batchSize: 32,
    shuffle: true,
    dropLast: true
);

// Валидационный датасет (можно другой стратегии)
Dataset valDataset = new LazyMNISTDataset(
    "t10k-images.idx3-ubyte",
    "t10k-labels.idx1-ubyte"
);

DataLoader valLoader = new SequentialDataLoader(valDataset, batchSize: 32);

// Используем в тренировке
foreach (var (features, labels) in trainLoader)
{
    // features: [32, 1, 28, 28] - батч из 32 картинок
    // labels: [32, 10] - one-hot метки
    model.Forward(features);
    // ...
}
```

## Почему такая архитектура правильная?

### 1. Разделение ответственности

- **Dataset** знает только о данных (как получить пример)
- **DataLoader** знает только об итерации (как сгруппировать в батчи)

Можно менять одно, не трогая другое. Хотите другой формат данных? Создайте новый Dataset. Хотите другую стратегию батчей? Создайте новый DataLoader.

### 2. Композиция вместо наследования

DataLoader принимает Dataset в конструкторе. Это позволяет комбинировать любые Dataset с любыми DataLoader.

### 3. Гибкость

Хотите перемешивать — берите DefaultDataLoader с shuffle=true. Не хотите — SequentialDataLoader. Хотите свой порядок — напишите свой DataLoader.

### 4. Производительность

LazyDataset позволяет работать с датасетами, не влезающими в память. DataLoader при этом может быть многопоточным (нужно добавить).

### 5. Тестируемость

Можно легко замокать Dataset и тестировать только логику DataLoader.

## Чего не хватает?

1. **Многопоточность** — в DefaultDataLoader можно добавить параллельную загрузку батчей
2. **Аугментация** — можно добавить трансформации данных на лету
3. **Prefetching** — загрузка следующего батча во время обработки текущего
4. **Разные форматы** — CSV, JSON, изображения с диска

## Заключение: Умные люди учатся на ошибках

Переосмыслив архитектуру, мы пришли к гораздо более чистой и гибкой системе. Теперь:

- **Dataset** — это просто провайдер отдельных примеров
- **DataLoader** — это итератор, собирающий эти примеры в батчи

Это разделение — не просто академическая красота. Оно позволяет:

- Загружать данные из разных источников (память, диск, сеть)
- Использовать разные стратегии батчей (перемешанные, последовательные, взвешенные)
- Легко добавлять новое поведение (аугментация, префетчинг)
- Тестировать компоненты независимо

Нет ничего страшного в том, чтобы переписать код, поняв, что первая версия была неидеальной. Страшно — оставить как есть и делать вид, что всё хорошо. Мы выбрали первый путь — путь постоянного улучшения.
