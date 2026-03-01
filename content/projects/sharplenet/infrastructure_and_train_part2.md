---
title: "SharpLeNet. Датасеты и Даталоадеры: Как накормить голодную нейросеть"
date: 2026-02-28
description: "SharpLeNet. Датасеты и Даталоадеры: Как накормить голодную нейросеть"
tags: ["SharpLeNet"]
categories: ["SharpLeNet"]
series: ["SharpLeNet"]
reading_time: 5
featured: true
---

Представьте, что вы шеф-повар в ресторане для нейросетей. Ваши клиенты (модели) очень требовательны: им нужно подавать данные порциями определенного размера, в случайном порядке, быстро и без задержек. Если вы замешкаетесь — нейросеть простаивает, GPU скучает, время теряется.

Сегодня мы заглянем на кухню и разберемся, как устроена система питания наших нейросетей — датасеты и даталоадеры.

## Датасет: Холодильник с продуктами

**Dataset** — это просто хранилище данных. Как холодильник: там лежат продукты (фичи) и этикетки с рецептами (метки). Никакой магии, просто структурированное хранение.

```csharp
public class Dataset : IDisposable, IEnumerable<(Tensor features, Tensor labels)>
{
    public Tensor Features { get; }  // Продукты
    public Tensor Labels { get; }    // Этикетки
    public int Size => Labels.Shape[0];  // Сколько всего блюд
```

**Что здесь важно?**

- `Features` — входные данные (картинки, текст, числа)
- `Labels` — правильные ответы (метки классов, целевые значения)
- `Size` — количество примеров в датасете

В конструкторе мы проверяем, что количество примеров в фичах и метках совпадает:

```csharp
if (features.Shape[0] != labels.Shape[0])
    throw new ArgumentException("Количество примеров должно совпадать");
```

Логично: если у вас 1000 картинок, то и меток должно быть 1000, а не 999.

## Индексы и перемешивание: Тайный порядок

Внутри датасета есть хитрый механизм — массив индексов:

```csharp
private int[] _indices;
private readonly int _batchSize;
private readonly bool _shuffle;
private readonly Random _rnd;
```

Зачем нужны индексы, если можно просто брать данные по порядку?

**Представьте:** У вас есть 1000 фотографий котов и собак, и они идут подряд: 500 котов, потом 500 собак. Если вы будете кормить сеть по порядку, то первые 500 шагов она будет видеть только котов и решит, что мир состоит из одних котов. Потом резко увидят 500 собак и запутаются.

Поэтому мы **перемешиваем** индексы:

```csharp
private void ShuffleIndices()
{
    for (int i = _indices.Length - 1; i > 0; i--)
    {
        int j = _rnd.Next(i + 1);
        (_indices[i], _indices[j]) = (_indices[j], _indices[i]);
    }
}
```

Это классический алгоритм Фишера-Йетса — быстрый и честный. Каждый пример имеет равные шансы оказаться на любой позиции.

## Итератор: Как датасет отдает данные порциями

Самое интересное — метод `GetEnumerator()`. Это сердце датасета, которое отвечает на вопрос: «Как нам итерироваться по данным батчами?»

```csharp
public IEnumerator<(Tensor features, Tensor labels)> GetEnumerator()
{
    int numSamples = Size;
    int featureSize = Features.Size / numSamples;  // Сколько чисел на один пример
    int labelSize = Labels.Size / numSamples;      // Сколько чисел на одну метку

    for (int startIdx = 0; startIdx < numSamples; startIdx += _batchSize)
    {
        int endIdx = Math.Min(startIdx + _batchSize, numSamples);
        int currentBatchSize = endIdx - startIdx;

        // Выделяем память под батч
        var batchFeaturesData = new double[currentBatchSize * featureSize];
        var batchLabelsData = new double[currentBatchSize * labelSize];

        // Копируем данные с учетом перемешанных индексов
        for (int i = 0; i < currentBatchSize; i++)
        {
            int sampleIdx = _indices[startIdx + i];  // ← вот тут магия!

            Array.Copy(Features.Data, sampleIdx * featureSize,
                      batchFeaturesData, i * featureSize, featureSize);
            Array.Copy(Labels.Data, sampleIdx * labelSize,
                      batchLabelsData, i * labelSize, labelSize);
        }

        // Создаем тензоры и отдаем
        var batchFeatures = new Tensor(batchFeaturesData,
            new int[] { currentBatchSize, featureSize });
        var batchLabels = new Tensor(batchLabelsData,
            new int[] { currentBatchSize, labelSize });

        yield return (batchFeatures, batchLabels);
    }
}
```

**Разберем по шагам:**

1. **Вычисляем размеры:** Узнаем, сколько чисел приходится на один пример фич и одну метку.

2. **Проходим по данным шагами batchSize:** `startIdx` — начало текущего батча.

3. **Обрабатываем последний батч:** Он может быть меньше остальных — `Math.Min` заботится об этом.

4. **Выделяем память** под текущий батч.

5. **Копируем данные с учетом перемешанных индексов:** Вот тут самое важное! Мы берем не `i`-й пример по порядку, а `_indices[startIdx + i]`-й. Благодаря этому данные идут в случайном порядке.

6. **Упаковываем в тензоры** и отдаем через `yield return`.

**Ключевой момент:** Данные физически копируются в новый массив для каждого батча. Это не очень эффективно (лишние копирования), но для учебной реализации — нормально. В продакшене используют буферы и zero-copy подходы.

## Сброс итератора: Начать заново

```csharp
public void Reset()
{
    _indices = Enumerable.Range(0, Size).ToArray();
    if (_shuffle)
    {
        ShuffleIndices();
    }
}
```

После каждой эпохи мы должны начать сначала, но с новым перемешиванием. `Reset()` создает новый массив индексов и, если нужно, перемешивает его.

## IDisposable: Уборка за собой

```csharp
protected virtual void Dispose(bool disposing)
{
    if (!_disposed)
    {
        if (disposing)
        {
            Features?.Dispose();
            Labels?.Dispose();
        }
        _disposed = true;
    }
}
```

Тензоры могут занимать много памяти (особенно на GPU). Поэтому мы реализуем `IDisposable`, чтобы можно было явно освободить ресурсы, когда датасет больше не нужен.

## Загрузчики данных: Разные источники, один интерфейс

Теперь посмотрим на интерфейс `IDataLoader`:

```csharp
public interface IDataLoader : IDisposable
{
    Dataset Load();
}
```

Этот интерфейс говорит: «Я умею загружать данные и превращать их в датасет». Откуда загружать — не важно. С диска, из памяти, из интернета, сгенерировать случайно — главное, вернуть `Dataset`.

### RandomDataLoader: Данные для экспериментов

```csharp
public class RandomDataLoader : IDataLoader
{
    public int NumSamples { get; set; }
    public int NumFeatures { get; set; }
    public int NumClasses { get; set; }

    public Dataset Load()
    {
        var rnd = new Random(42);

        // Случайные фичи в диапазоне [-1, 1]
        var featuresData = new double[NumSamples * NumFeatures];
        for (int i = 0; i < featuresData.Length; i++)
        {
            featuresData[i] = rnd.NextDouble() * 2 - 1;
        }
        var features = new Tensor(featuresData, new int[] { NumSamples, NumFeatures });

        // One-hot метки
        var labelsData = new double[NumSamples * NumClasses];
        for (int i = 0; i < NumSamples; i++)
        {
            int classIdx = rnd.Next(NumClasses);
            labelsData[i * NumClasses + classIdx] = 1.0;
        }
        var labels = new Tensor(labelsData, new int[] { NumSamples, NumClasses });

        return new Dataset(features, labels);
    }
}
```

Этот загрузчик генерирует случайные данные. Зачем? Для тестирования! Прежде чем грузить реальный MNIST, можно проверить, что сеть вообще способна обучаться на случайных данных (спойлер: не должна, если все правильно, loss должен оставаться высоким).

**Особенности:**

- Фичи в диапазоне [-1, 1] — хорошая практика для нормализации
- One-hot метки — то, что ждет CrossEntropyLoss
- Фиксированный seed (42) — для воспроизводимости

### MNISTDataLoader: Заглушка для будущего

```csharp
public class MNISTDataLoader : IDataLoader
{
    public Dataset Load()
    {
        throw new NotImplementedException();
    }
}
```

Пока не реализовано, но интерфейс уже готов. Когда-нибудь здесь будет загрузка реальных картинок с цифрами.

## Как это все работает вместе?

Давайте посмотрим на типичный сценарий использования:

```csharp
// Создаем загрузчик
var loader = new RandomDataLoader(1000, 784, 10);  // 1000 примеров, 784 фичи, 10 классов

// Загружаем датасет
using var dataset = loader.Load();  // using гарантирует освобождение памяти

// Итерируемся по батчам
foreach (var batch in dataset)
{
    // batch.features: [32, 784] — батч из 32 картинок 28x28
    // batch.labels: [32, 10] — one-hot метки для 32 примеров

    // Тут будет прямой проход, loss, backward и т.д.
    Console.WriteLine($"Батч: фичи {batch.features.Shape}, метки {batch.labels.Shape}");
}

// Можно начать новую эпоху с новым перемешиванием
dataset.Reset();
foreach (var batch in dataset)  // Данные пойдут в другом порядке
{
    // Обработка второй эпохи
}
```

## Архитектурные решения: Почему так, а не иначе?

### 1. Датасет сам итерируется по батчам

В некоторых фреймворках есть отдельный класс DataLoader, который принимает Dataset и разбивает на батчи. Мы объединили их в одном классе для простоты. Плюс: меньше кода. Минус: меньше гибкости.

### 2. Копирование данных при каждом батче

Да, это неэффективно. Но:

- Упрощает код
- Хорошо для обучения (данные все равно маленькие)
- Легко понять

В реальных системах используют буферы и асинхронную загрузку.

### 3. IDisposable на тензорах

Тензоры могут жить на GPU. Если не освобождать память явно, GC может не справиться. Поэтому мы реализуем `Dispose` и в датасете, и в загрузчиках.

### 4. Интерфейс IDataLoader

Позволяет подменять источники данных без изменения остального кода. Хотите MNIST — пожалуйста, хотите CIFAR — легко, хотите свой кастомный формат — реализуйте интерфейс.

## Чего не хватает?

1. **Асинхронности** — загрузка следующего батча во время обучения текущего
2. **Аугментации** — преобразования данных на лету
3. **Prefetching** — подготовка данных заранее
4. **Multi-threading** — загрузка в несколько потоков
5. **Кэширования** — чтобы не читать с диска каждый раз

Но для учебной реализации — то, что нужно. Понятно, работает, можно расширять.

## Заключение: Ресторан открыт

Теперь у нас есть полноценная система питания для нейросетей:

- **Dataset** — холодильник с продуктами
- **Индексы и перемешивание** — чтобы не кормить одной и той же едой
- **Батчи** — порции правильного размера
- **Загрузчики** — разные источники продуктов
- **Dispose** — уборка на кухне
