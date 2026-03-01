---
title: "SharpLeNet. Метрики и Колбэки: Как понять, что нейросеть не дурачится"
date: 2026-02-28
description: "SharpLeNet. Метрики и Колбэки: Как понять, что нейросеть не дурачится"
tags: ["SharpLeNet"]
categories: ["SharpLeNet"]
series: ["SharpLeNet"]
reading_time: 5
featured: true
---

Друзья, мы построили нейросеть, научились кормить ее данными, считать ошибки и обновлять веса. Но остаются два важных вопроса: **«Как понять, хорошо ли учится наша сеть?»** и **«Как вмешаться в процесс, если что-то идет не так?»**

Ответы — метрики и колбэки. Метрики — это наша система оценки, которая говорит правду, не приукрашивая. Колбэки — это наши помощники, которые следят за процессом и реагируют на события.

## Метрики: Честная оценка без прикрас

Функция потерь (loss) говорит нам, как учиться. Но для оценки реального качества нужны метрики. Почему? Потому что loss часто непонятен человеку. Что значит «кросс-энтропия 0.3»? А «accuracy 0.87» — это понятно: 87% правильных ответов.

### Accuracy: Простой и понятный

```csharp
public static double Accuracy(Tensor predictions, Tensor targets)
{
    if (predictions.Rank != 2 || targets.Rank != 2)
        throw new ArgumentException("Ожидаются двумерные тензоры [batch, classes]");

    int batchSize = predictions.Shape[0];
    int numClasses = predictions.Shape[1];

    int correct = 0;
    for (int i = 0; i < batchSize; i++)
    {
        // Находим предсказанный класс (максимальная вероятность)
        int predClass = 0;
        double maxProb = double.MinValue;
        for (int j = 0; j < numClasses; j++)
        {
            if (predictions[i, j] > maxProb)
            {
                maxProb = predictions[i, j];
                predClass = j;
            }
        }

        // Находим истинный класс (где 1.0 в one-hot)
        int trueClass = 0;
        for (int j = 0; j < numClasses; j++)
        {
            if (Math.Abs(targets[i, j] - 1.0) < 1e-6)
            {
                trueClass = j;
                break;
            }
        }

        if (predClass == trueClass)
            correct++;
    }

    return (double)correct / batchSize;
}
```

**Как это работает:**

1. Для каждого примера в батче находим класс с максимальной вероятностью (argmax).
2. Находим истинный класс (где в one-hot стоит 1.0).
3. Сравниваем.
4. Делим количество правильных ответов на размер батча.

**Пример:**

```csharp
// Предсказания для 3 примеров, 4 класса
predictions = [
    [0.1, 0.7, 0.1, 0.1],  // предсказан класс 1
    [0.4, 0.3, 0.2, 0.1],  // предсказан класс 0
    [0.1, 0.1, 0.1, 0.7]   // предсказан класс 3
];

targets = [
    [0, 1, 0, 0],  // истина: класс 1 ✓
    [1, 0, 0, 0],  // истина: класс 0 ✓
    [0, 0, 1, 0]   // истина: класс 2 ✗ (должен быть 2, а предсказан 3)
];

// Accuracy = 2 / 3 = 0.6667
```

**Важный момент:** Мы сравниваем с 1.0 с допуском `1e-6`. Почему? Потому что из-за вычислений с плавающей точкой 1.0 может стать 0.9999999. Маленький допуск спасает от ложных ошибок.

### MeanLoss: Средняя температура по больнице

```csharp
public static double MeanLoss(List<double> losses)
{
    return losses.Count > 0 ? losses.Average() : 0.0;
}
```

Эта метрика собирает значения loss за эпоху и усредняет их. Полезна для отслеживания тренда: loss падает — значит, учимся.

### ProgressBar: Красивый вывод

```csharp
public static void PrintProgress(int epoch, int totalEpochs,
    double trainLoss, double trainAcc, double valLoss, double valAcc)
{
    Console.WriteLine(
        $"Epoch [{epoch + 1}/{totalEpochs}] | " +
        $"Train Loss: {trainLoss:F6} | Train Acc: {trainAcc:F4} | " +
        $"Val Loss: {valLoss:F6} | Val Acc: {valAcc:F4}");
}
```

Выводит аккуратную строчку с результатами эпохи. `:F6` — 6 знаков после запятой для loss, `:F4` — 4 знака для accuracy. Так и читать приятно, и числа не расползаются.

## Колбэки: Шпионы в мире обучения

Колбэки (callbacks) — это функции, которые вызываются в определенные моменты обучения. Они как шпионы: сидят тихо, наблюдают, но в нужный момент вмешиваются.

### Базовый класс: Контракт для всех колбэков

```csharp
public abstract class Callback
{
    public virtual void OnTrainBegin() { }
    public virtual void OnTrainEnd() { }
    public virtual void OnEpochBegin(int epoch) { }
    public virtual void OnEpochEnd(int epoch, double trainLoss, double valLoss,
        double trainAcc, double valAcc) { }
    public virtual void OnBatchEnd(int batch, double loss) { }
}
```

Это как интерфейс с пустыми реализациями. Каждый колбэк может переопределить только те методы, которые ему нужны. Остальные просто ничего не делают.

**Точки входа:**

- `OnTrainBegin` / `OnTrainEnd` — в начале и конце всего обучения
- `OnEpochBegin` / `OnEpochEnd` — в начале и конце каждой эпохи
- `OnBatchEnd` — после каждого батча

### EarlyStopping: Вовремя остановиться

Представьте, что вы учите студента. Сначала он быстро прогрессирует, потом медленнее, а потом начинает зубрить и перестает понимать суть. Хороший преподаватель знает, когда пора остановиться.

`EarlyStopping` — это такой преподаватель для нейросети:

```csharp
public class EarlyStopping : Callback
{
    private readonly int _patience;
    private readonly double _minDelta;
    private int _wait;
    private double _bestLoss;
    private bool _stopTraining;

    public EarlyStopping(ILogger<EarlyStopping> logger, int patience = 5, double minDelate = 1e-4)
    {
        _patience = patience;
        _minDelta = minDelate;
        _wait = 0;
        _stopTraining = false;
        _bestLoss = double.MaxValue;
    }

    public override void OnEpochEnd(int epoch, double trainLoss, double valLoss,
        double trainAcc, double valAcc)
    {
        if (valLoss < _bestLoss - _minDelta)
        {
            // Улучшение есть
            _bestLoss = valLoss;
            _wait = 0;
        }
        else
        {
            // Улучшения нет
            _wait++;
            if (_wait > _patience)
            {
                Console.WriteLine($"\nРанняя остановка на эпохе {epoch + 1}");
                _stopTraining = true;
            }
        }
    }

    public bool ShouldStop => _stopTraining;
}
```

**Как работает:**

1. Следим за валидационным loss (`valLoss`).
2. Если он уменьшился хотя бы на `_minDelta` — радуемся и сбрасываем счетчик.
3. Если не уменьшался `_patience` эпох подряд — останавливаем обучение.

**Параметры:**

- `patience = 5` — сколько эпох ждать улучшения
- `minDelta = 1e-4` — минимальное улучшение, которое считаем значимым

**Аналогия:** Вы ждете автобус. Если он не приходит 5 минут (patience) — вы начинаете волноваться. Если не приходит 5 минут подряд — вы уходите (останавливаете обучение).

### ModelCheckpoint: Сохранять лучшее

Бывает, что нейросеть нашла отличные веса на 10-й эпохе, а потом переобучилась и испортилась. `ModelCheckpoint` сохраняет лучшую версию:

```csharp
public class ModelCheckpoint : Callback
{
    private readonly Model _model;
    private readonly string _filePath;
    private double _bestLoss;

    public ModelCheckpoint(Model model, string filePath)
    {
        _model = model;
        _filePath = filePath;
        _bestLoss = double.MaxValue;
    }

    public override void OnEpochEnd(int epoch, double trainLoss, double valLoss,
        double trainAcc, double valAcc)
    {
        if (valLoss < _bestLoss)
        {
            _bestLoss = valLoss;
            SaveModel($"epoch_{epoch + 1}_loss_{valLoss:F6}.bin");
            Console.WriteLine($"  ✓ Модель сохранена (лучшая val_loss: {valLoss:F6})");
        }
    }

    private void SaveModel(string fileName)
    {
        // TODO: Реализовать сериализацию модели
        Console.WriteLine($"  Сохранение в {fileName}");
    }
}
```

**Что происходит:**

- На каждой эпохе проверяем, улучшился ли валидационный loss
- Если да — сохраняем модель
- В имени файла указываем эпоху и значение loss (чтобы потом легко найти)

**Важно:** `SaveModel` пока не реализован (TODO). В реальной системе здесь была бы сериализация весов модели в файл.

## Как это работает вместе?

Давайте представим типичный тренировочный цикл с метриками и колбэками:

```csharp
public class Trainer
{
    private readonly List<Callback> _callbacks;
    private readonly List<double> _trainLosses = new();
    private readonly List<double> _valLosses = new();

    public void Train(int epochs)
    {
        // Начало обучения
        foreach (var cb in _callbacks) cb.OnTrainBegin();

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            // Начало эпохи
            foreach (var cb in _callbacks) cb.OnEpochBegin(epoch);

            // Тренировка
            double trainLoss = TrainEpoch();
            _trainLosses.Add(trainLoss);

            // Валидация
            var (valLoss, valAcc) = Validate();
            _valLosses.Add(valLoss);

            // Собираем тренировочные метрики
            double trainAcc = CalculateTrainAccuracy();

            // Конец эпохи
            foreach (var cb in _callbacks)
                cb.OnEpochEnd(epoch, trainLoss, valLoss, trainAcc, valAcc);

            // Печатаем прогресс
            Metrics.PrintProgress(epoch, epochs, trainLoss, trainAcc, valLoss, valAcc);

            // Проверяем раннюю остановку
            if (_callbacks.OfType<EarlyStopping>().Any(es => es.ShouldStop))
                break;
        }

        // Конец обучения
        foreach (var cb in _callbacks) cb.OnTrainEnd();
    }

    private double TrainEpoch()
    {
        double epochLoss = 0;
        int batches = 0;

        foreach (var batch in _trainLoader)
        {
            // Прямой проход, loss, backward, step...
            double batchLoss = ProcessBatch(batch);

            epochLoss += batchLoss;
            batches++;

            // Колбэк после каждого батча
            foreach (var cb in _callbacks)
                cb.OnBatchEnd(batches, batchLoss);
        }

        return Metrics.MeanLoss(new List<double> { epochLoss / batches });
    }
}
```

## Метрики vs Loss: В чем разница?

Часто путают эти понятия. Давайте расставим точки над i:

|                        | Loss                                  | Метрики                       |
| ---------------------- | ------------------------------------- | ----------------------------- |
| **Цель**               | Учить сеть (обратное распространение) | Оценивать сеть (для человека) |
| **Интерпретация**      | Часто непонятна (0.3 — это много?)    | Понятна (87% — это хорошо)    |
| **Дифференцируемость** | Обязана быть                          | Не обязана                    |
| **Использование**      | В оптимизаторе                        | В отчетах, ранней остановке   |
| **Пример**             | CrossEntropyLoss                      | Accuracy, Precision, F1       |

## Что дальше?

В нашей реализации есть основа, но можно добавить:

1. **Больше метрик** — Precision, Recall, F1, Confusion Matrix
2. **Больше колбэков** — LearningRateScheduler, TensorBoardLogger, CSVLogger
3. **Асинхронность** — чтобы колбэки не тормозили обучение
4. **Приоритеты** — чтобы одни колбэки вызывались до других
5. **Контекст** — передавать больше информации в колбэки

## Заключение: Невидимые помощники

Метрики и колбэки — это как приборная панель и автопилот в самолете. Метрики показывают высоту, скорость, курс (как учится сеть). Колбэки автоматически реагируют на показания: если что-то идет не так — включают сигнализацию (early stopping) или записывают данные в черный ящик (model checkpoint).

Без них обучение было бы слепым полетом. Вы бы видели только loss на последнем батче и гадали: "Ну как там вообще дела? Улучшается или нет? Может, уже хватит учить?"

С метриками и колбэками у вас есть полный контроль. Вы видите точную картину. Вы можете вмешаться, когда нужно. И вы точно знаете, когда пора остановиться.
