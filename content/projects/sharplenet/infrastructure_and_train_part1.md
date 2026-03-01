---
title: "SharpLeNet. Жизненный цикл обучения: Как нейросеть превращается из ученика в профессионала"
date: 2026-02-28
description: "SharpLeNet. Жизненный цикл обучения: Как нейросеть превращается из ученика в профессионала"
tags: ["SharpLeNet"]
categories: ["SharpLeNet"]
series: ["SharpLeNet"]
reading_time: 5
featured: true
---

Друзья, мы прошли долгий путь. Мы разобрали слои, функции активации, функции потерь, оптимизаторы. У нас есть все кирпичики для построения нейросети. Но остался самый важный вопрос: **как из этих кирпичиков построить работающую систему?**

Представьте, что у вас есть отличный двигатель, хорошие колеса, руль и сиденья. Но это еще не машина. Нужна рама, которая соединит все вместе, нужна проводка, нужна приборная панель. В мире нейросетей эту роль играет "обвязка" — загрузчики данных, метрики, колбэки и сам процесс тренировки.

## Загрузчик данных: Конвейер, который кормит нейросеть

Нейросеть голодна. Она хочет данных, много данных, постоянно. Но просто скормить ей весь датасет сразу — плохая идея. Это как скормить человеку недельный запас еды за один присест. Ничего хорошего не выйдет.

### Проблемы, которые решает загрузчик данных:

1. **Память** — весь датасет может не поместиться в RAM/GPU
2. **Скорость** — нужно кормить сеть быстрее, чем она учится
3. **Перемешивание** — данные должны подаваться в случайном порядке
4. **Баланс классов** — чтобы сеть не "перекосило" в сторону частых классов

### Как это работает:

```csharp
// Создаем датасет (просто коллекция пар вход-выход)
var dataset = new ImageDataset("путь/к/картинкам");

// Оборачиваем в загрузчик с батчами по 32
var loader = new DataLoader(dataset, batchSize: 32, shuffle: true);

// Теперь можно итерироваться
foreach (var batch in loader)
{
    // batch — это мини-батч из 32 картинок
    // Прямой проход, обратный проход, шаг оптимизатора
}
```

**Аналогия:** Загрузчик данных — это как конвейер на фабрике. Рабочие (нейросеть) берут детали (данные) с конвейера, обрабатывают их, а конвейер подвозит новые. Конвейер работает с оптимальной скоростью, детали перемешаны, чтобы не было привыкания к порядку.

## Метрики: Система оценки, которая не обманывает

Функция потерь говорит нам, как учиться. Но для оценки реального качества нужны **метрики**. Почему нельзя просто смотреть на loss?

**Проблема loss:**

- Loss может уменьшаться, а качество — падать (например, при дисбалансе классов)
- Loss часто неинтерпретируем для человека (что такое "энтропия 0.3"?)
- Разные loss-функции дают числа в разных масштабах

**Метрики говорят на человеческом языке:**

### Accuracy (точность)

```csharp
public class Accuracy : Metric
{
    public override double Calculate(Tensor predictions, Tensor targets)
    {
        int correct = 0;
        int total = predictions.Shape[0];

        for (int i = 0; i < total; i++)
        {
            int predClass = ArgMax(predictions[i]);
            int trueClass = ArgMax(targets[i]);
            if (predClass == trueClass) correct++;
        }

        return (double)correct / total;
    }
}
```

Процент правильных ответов. Просто и понятно. "Модель угадала в 87% случаев".

### Precision (точность) и Recall (полнота)

Для задач, где важны разные типы ошибок:

- **Precision:** из того, что модель посчитала котами, сколько реально котов?
- **Recall:** из всех реальных котов, сколько модель нашла?

### F1-score

Гармоническое среднее precision и recall. Когда нужно учесть оба.

### Confusion Matrix

Таблица, показывающая, какие классы с какими путаются.

**Аналогия:** Loss — это как внутреннее ощущение ученика ("кажется, я ошибся"). Метрики — это оценка на экзамене ("твердая четверка"). Одно субъективно и нужно для обучения, другое объективно и нужно для оценки.

## Колбэки: Шпионы в стане врага

Колбэки (callbacks) — это функции, которые вызываются в определенные моменты обучения. Они как шпионы, которые следят за процессом и вмешиваются, когда нужно.

### Зачем нужны колбэки?

```csharp
// Типичный колбэк для сохранения лучшей модели
public class ModelCheckpoint : Callback
{
    private double bestLoss = double.MaxValue;
    private string savePath;

    public override void OnEpochEnd(Loss loss, List<Metric> metrics)
    {
        if (loss.Value < bestLoss)
        {
            bestLoss = loss.Value;
            SaveModel(savePath);
            Console.WriteLine($"Модель сохранена! Loss: {bestLoss}");
        }
    }
}
```

### Основные типы колбэков:

1. **ModelCheckpoint** — сохраняет модель, когда она становится лучше
2. **EarlyStopping** — останавливает обучение, если метрики перестали улучшаться
3. **LearningRateScheduler** — меняет learning rate по расписанию
4. **TensorBoard/Logger** — логирует метрики для визуализации
5. **ProgressBar** — показывает красивый прогресс-бар

### Типичные точки для колбэков:

- `OnBatchStart` / `OnBatchEnd` — до и после обработки батча
- `OnEpochStart` / `OnEpochEnd` — до и после эпохи
- `OnTrainStart` / `OnTrainEnd` — в начале и конце обучения
- `OnValidationStart` / `OnValidationEnd` — до и после валидации

**Аналогия:** Колбэки — это как ассистенты режиссера на съемочной площадке. Один следит за временем, другой — за бюджетом, третий — за качеством материала. Они не снимают кино, но без них процесс развалится.

## Процесс тренировки: Собираем все вместе

А теперь давайте посмотрим, как все эти компоненты работают вместе:

```csharp
public class Trainer
{
    private readonly Model _model;
    private readonly DataLoader _trainLoader;
    private readonly DataLoader _valLoader;
    private readonly Loss _loss;
    private readonly Optimizer _optimizer;
    private readonly List<Metric> _metrics;
    private readonly List<Callback> _callbacks;

    public void Train(int epochs)
    {
        foreach (var callback in _callbacks) callback.OnTrainStart();

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            foreach (var callback in _callbacks) callback.OnEpochStart(epoch);

            // Тренировочная фаза
            _model.Train();  // Переключаем модель в режим обучения
            double trainLoss = TrainEpoch();

            // Валидационная фаза
            _model.Eval();   // Переключаем модель в режим оценки
            var valMetrics = Validate();

            foreach (var callback in _callbacks)
                callback.OnEpochEnd(trainLoss, valMetrics);

            // Проверяем, не пора ли остановиться
            if (ShouldStopEarly()) break;
        }

        foreach (var callback in _callbacks) callback.OnTrainEnd();
    }

    private double TrainEpoch()
    {
        double epochLoss = 0;
        int batches = 0;

        foreach (var batch in _trainLoader)
        {
            foreach (var callback in _callbacks) callback.OnBatchStart();

            // 1. Прямой проход
            var predictions = _model.Forward(batch.Input);

            // 2. Вычисление loss
            var loss = _loss.Compute(predictions, batch.Targets);

            // 3. Обратный проход
            loss.Backward();

            // 4. Шаг оптимизатора
            _optimizer.Step();

            // 5. Обнуление градиентов
            _optimizer.ZeroGrad();

            epochLoss += loss.Value;
            batches++;

            foreach (var callback in _callbacks) callback.OnBatchEnd(loss);
        }

        return epochLoss / batches;
    }

    private Dictionary<string, double> Validate()
    {
        var results = new Dictionary<string, double>();

        // Инициализируем метрики
        foreach (var metric in _metrics)
            metric.Reset();

        using (NoGrad())  // Отключаем вычисление градиентов
        {
            foreach (var batch in _valLoader)
            {
                var predictions = _model.Forward(batch.Input);

                // Обновляем все метрики
                foreach (var metric in _metrics)
                    metric.Update(predictions, batch.Targets);
            }
        }

        // Собираем результаты
        foreach (var metric in _metrics)
            results[metric.Name] = metric.Calculate();

        return results;
    }
}
```

### Ключевые моменты:

1. **Режимы модели** — `Train()` и `Eval()` переключают поведение слоев (Dropout, BatchNorm работают по-разному)

2. **Разделение данных** — тренировочные данные для обучения, валидационные для честной оценки

3. **Градиенты только на тренировке** — на валидации мы отключаем autograd (`NoGrad()`), чтобы не тратить память

4. **Метрики накапливаются** — accuracy считается по всем батчам валидации, а не усредняется

## Жизненный цикл одной эпохи

Давайте проследим, что происходит в течение одной эпохи:

```
НАЧАЛО ЭПОХИ
    ↓
Колбэк: OnEpochStart
    ↓
[ТРЕНИРОВКА]
    Для каждого батча:
        Колбэк: OnBatchStart
        Прямой проход → Loss → Backward → Step → ZeroGrad
        Колбэк: OnBatchEnd
    ↓
[ВАЛИДАЦИЯ]
    Отключаем градиенты
    Для каждого батча:
        Прямой проход
        Обновление метрик
    ↓
Колбэк: OnEpochEnd (с результатами)
    ↓
Проверка EarlyStopping
    ↓
КОНЕЦ ЭПОХИ (или переход к следующей)
```

## Инфраструктура обучения: Что еще нужно?

### 1. Логирование

```csharp
public class Logger : Callback
{
    public override void OnEpochEnd(double loss, Dictionary<string, double> metrics)
    {
        Console.WriteLine($"Epoch {CurrentEpoch}: Loss = {loss:F4}");
        foreach (var (name, value) in metrics)
            Console.WriteLine($"  {name}: {value:F4}");

        // Можно писать в файл, в TensorBoard, в базу данных
        TensorBoard.AddScalar("train/loss", loss, CurrentEpoch);
    }
}
```

### 2. Ранняя остановка (Early Stopping)

```csharp
public class EarlyStopping : Callback
{
    private int _patience;
    private int _badEpochs = 0;
    private double _bestMetric = double.MinValue;

    public override void OnEpochEnd(Dictionary<string, double> metrics)
    {
        double current = metrics["val_accuracy"];

        if (current > _bestMetric)
        {
            _bestMetric = current;
            _badEpochs = 0;
        }
        else
        {
            _badEpochs++;
            if (_badEpochs >= _patience)
            {
                StopTraining();  // Сигнал остановить обучение
            }
        }
    }
}
```

### 3. Планировщик learning rate

```csharp
public class StepLR : Callback
{
    private int _stepSize;
    private double _gamma;

    public override void OnEpochEnd()
    {
        if (CurrentEpoch % _stepSize == 0)
        {
            Optimizer.LearningRate *= _gamma;
            Console.WriteLine($"Learning rate уменьшен до {Optimizer.LearningRate}");
        }
    }
}
```

### 4. Прогресс-бар

```csharp
public class ProgressBar : Callback
{
    private int _totalBatches;
    private int _currentBatch;
    private DateTime _startTime;

    public override void OnEpochStart()
    {
        _totalBatches = TrainLoader.Length;
        _currentBatch = 0;
        _startTime = DateTime.Now;
    }

    public override void OnBatchEnd(Loss loss)
    {
        _currentBatch++;
        double progress = (double)_currentBatch / _totalBatches;
        double elapsed = (DateTime.Now - _startTime).TotalSeconds;
        double eta = elapsed / progress - elapsed;

        Console.Write($"\r[{new string('#', (int)(progress*50))}" +
                     $"{new string('-', 50-(int)(progress*50))}] " +
                     $"{progress*100:F1}% | Loss: {loss.Value:F4} | ETA: {eta:F0}s");
    }
}
```

## Чего мы не коснулись (пока)

В следующих статьях мы подробно разберем:

1. **DataLoader** — как грузить данные
2. **Метрики** — реализации различных метрик
3. **Колбэки** — полноценная система событий с приоритетами и контекстом
4. **Trainer** — гибкая система обучения
5. **Валидация** — как правильно оценивать модели без переобучения

## Заключение: Машина готова к поездке

Теперь у нас есть не просто набор слоев и функций, а полноценная система для обучения нейросетей. У нас есть:

- **Конвейер** (DataLoader), который кормит сеть
- **Двигатель** (модель), который преобразует входы в выходы
- **Учитель** (Loss), который оценивает ошибки
- **Тренер** (Optimizer), который исправляет ошибки
- **Экзаменаторы** (Metrics), которые ставят объективные оценки
- **Ассистенты** (Callbacks), которые следят за процессом

Вместе они создают единый организм, способный учиться на данных, адаптироваться к задачам и становиться лучше с каждой эпохой.

В следующих статьях мы углубимся в каждую из этих подсистем и реализуем их с нуля.
