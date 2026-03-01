---
title: "SharpLeNet. Тренер: Дирижер оркестра нейросети"
date: 2026-02-28
description: "SharpLeNet. Тренер: Дирижер оркестра нейросети"
tags: ["SharpLeNet"]
categories: ["SharpLeNet"]
series: ["SharpLeNet"]
reading_time: 5
featured: true
---

Друзья, мы собрали все ингредиенты: у нас есть модель (слои), есть функция потерь (учитель), есть оптимизатор (тренер), есть данные (еда) и есть метрики с колбэками (система контроля). Осталось собрать это в единый работающий механизм. Встречайте — **Trainer**, главный дирижер нашего оркестра.

## Роль тренера: Почему без него никак?

Представьте оркестр. У вас есть скрипачи (слои), дирижерская палочка (оптимизатор), партитура (функция потерь) и ноты (данные). Но кто скажет скрипачам, когда начинать, когда заканчивать, когда играть тише, а когда громче? Кто следит, чтобы все играли слаженно?

В мире нейросетей эту роль играет **Trainer**. Это главный координатор, который:

1. Запускает эпохи обучения
2. Перебирает батчи данных
3. Вызывает прямой и обратный проходы
4. Собирает метрики
5. Уведомляет колбэки о событиях
6. Решает, когда остановиться

## Структура Trainer: Что у него внутри?

```csharp
public class Trainer
{
    private readonly Model _model;              // Что учим
    private readonly Optimizer _optimizer;      // Как учим
    private readonly Loss _loss;                 // Чем измеряем ошибки
    private readonly Dataset _trainDataset;      // На чем учим
    private readonly Dataset? _valDataset;       // На чем проверяем
    private readonly List<Callback> _callbacks;  // Кто следит за процессом

    private List<double> _trainLosses;           // История обучения
    private List<double> _valLosses;             // История валидации
    private List<double> _trainAccuracies;       // Точность на тренировке
    private List<double> _valAccuracies;         // Точность на валидации

    public IReadOnlyCollection<double> TrainLosses => _trainLosses.AsReadOnly();
    public IReadOnlyCollection<double> ValLosses => _valLosses.AsReadOnly();
```

**Разберем по частям:**

- `_model` — та самая нейросеть, которую мы собираемся учить
- `_optimizer` — алгоритм обновления весов (SGD, Adam и т.д.)
- `_loss` — функция потерь (MSE, BCE, CrossEntropy)
- `_trainDataset` — данные для обучения
- `_valDataset` — данные для валидации (может быть null)
- `_callbacks` — список наблюдателей за процессом
- История в `List<double>` — чтобы потом строить графики

## Конструктор: Сборка команды

```csharp
public Trainer(Model model, Optimizer optimizer, Loss loss,
    Dataset trainDataset, Dataset? valDataset = null)
{
    _model = model;
    _optimizer = optimizer;
    _loss = loss;
    _trainDataset = trainDataset;
    _valDataset = valDataset;
    _callbacks = new List<Callback>();
    _trainLosses = new List<double>();
    _valLosses = new List<double>();
    _trainAccuracies = new List<double>();
    _valAccuracies = new List<double>();
}
```

Конструктор принимает все необходимое и инициализирует внутренние структуры. Валидационный датасет опционален — можно учить и без проверки, но тогда мы не узнаем, не переобучилась ли сеть.

## Добавление колбэков: Расширяем функциональность

```csharp
public void AddCallback(Callback callback)
{
    _callbacks.Add(callback);
}
```

Простой метод, но он открывает безграничные возможности. Хотите сохранять лучшую модель? Добавьте `ModelCheckpoint`. Хотите останавливаться при отсутствии прогресса? Добавьте `EarlyStopping`. Хотите логировать в TensorBoard? Создайте свой колбэк и добавьте его.

## Тренировочная эпоха: Сердце обучения

```csharp
private TrainingStats TrainEpoch()
{
    double totalLoss = 0;
    double totalAcc = 0;
    int batches = 0;

    foreach (var (batchFeatures, batchLabels) in _trainDataset)
    {
        // 1. Прямой проход
        Tensor predictions = _model.Forward(batchFeatures);
        Tensor loss = _loss.Compute(predictions, batchLabels);

        // 2. Считаем метрики
        double lossValue = loss.Data[0];
        double accValue = Metrics.Metrics.Accuracy(predictions, batchLabels);

        totalLoss += lossValue;
        totalAcc += accValue;
        batches++;

        // 3. Обратный проход
        _optimizer.ZeroGrad();     // Обнуляем старые градиенты
        loss.Backward();            // Считаем новые градиенты
        _optimizer.Step();          // Обновляем веса

        // 4. Уведомляем колбэки
        foreach (Callback callback in _callbacks)
        {
            callback.OnBatchEnd(batches, lossValue);
        }
    }

    return new TrainingStats
    {
        Loss = totalLoss / batches,
        Accuracy = totalAcc / batches,
    };
}
```

**Давайте пройдем по шагам:**

### Шаг 1: Прямой проход

Пропускаем батч через модель, получаем предсказания. Сравниваем предсказания с истинными метками через функцию потерь. Получаем тензор loss (обычно скаляр).

### Шаг 2: Сбор метрик

Извлекаем числовое значение loss (`loss.Data[0]`) и вычисляем accuracy. Копим суммы для последующего усреднения.

### Шаг 3: Обратный проход

Классическая триада оптимизации:

- `ZeroGrad()` — очищаем старые градиенты, чтобы они не накапливались
- `Backward()` — вычисляем новые градиенты по всему графу вычислений
- `Step()` — обновляем веса согласно градиентам и выбранному оптимизатору

### Шаг 4: Колбэки

Сообщаем всем заинтересованным, что батч обработан. Можно логировать прогресс, обновлять прогресс-бар, считать скользящее среднее.

### Возврат статистики

Усредняем loss и accuracy по всем батчам эпохи. Это дает нам общую картину: как модель выступила на этой эпохе.

## Валидационная эпоха: Честная проверка

```csharp
private TrainingStats ValidateEpoch()
{
    if (_valDataset == null)
        return new TrainingStats { Loss = 0, Accuracy = 0 };

    double totalLoss = 0;
    double totalAcc = 0;
    int batches = 0;

    foreach (var (batchFeatures, batchLabels) in _valDataset)
    {
        Tensor predictions = _model.Forward(batchFeatures);
        Tensor loss = _loss.Compute(predictions, batchLabels);
        double lossValue = loss.Data[0];

        totalLoss += lossValue;
        totalAcc += Metrics.Metrics.Accuracy(predictions, batchLabels);
        batches++;
    }

    return new TrainingStats
    {
        Loss = totalLoss / batches,
        Accuracy = totalAcc / batches
    };
}
```

**Ключевое отличие от тренировки:** Здесь нет обратного прохода! Мы только считаем предсказания и метрики, но не обновляем веса. И никаких `ZeroGrad`, `Backward`, `Step`.

Почему? Потому что валидация — это экзамен. На экзамене нельзя подсматривать в учебник и нельзя менять ответы после проверки. Мы просто смотрим, как модель справляется с новыми, невиданными ранее данными.

## Fit: Главный метод обучения

```csharp
public void Fit(int epochs)
{
    Console.WriteLine("=== Начало обучения ===");
    Console.WriteLine($"Модель: {_model.GetType().Name}");
    Console.WriteLine($"Оптимизатор: {_optimizer.GetType().Name}");
    Console.WriteLine($"Функция потерь: {_loss.GetType().Name}");
    Console.WriteLine($"Эпох: {epochs}");
    Console.WriteLine($"Train samples: {_trainDataset.Count() * _trainDataset.First().features.Shape[0]}");
    Console.WriteLine();

    // Начало обучения
    foreach (Callback callback in _callbacks)
        callback.OnTrainBegin();

    bool isEarlyStopping = false;
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        // Начало эпохи
        foreach (Callback callback in _callbacks)
            callback.OnEpochBegin(epoch);

        // Обучение
        TrainingStats trainingStats = TrainEpoch();
        _trainLosses.Add(trainingStats.Loss);
        _trainAccuracies.Add(trainingStats.Accuracy);

        // Валидация
        TrainingStats valStats = ValidateEpoch();
        _valLosses.Add(valStats.Loss);
        _valAccuracies.Add(valStats.Accuracy);

        // Вывод прогресса
        Metrics.Metrics.PrintProgress(epoch, epochs, trainingStats.Loss, trainingStats.Accuracy,
            valStats.Loss, valStats.Accuracy);

        foreach (Callback callback in _callbacks)
        {
            callback.OnEpochEnd(epoch, trainingStats.Loss, valStats.Loss,
                trainingStats.Accuracy, valStats.Accuracy);

            // Проверка на раннюю остановку
            if (callback is EarlyStopping es && es.ShouldStop)
            {
                Console.WriteLine("Обучение остановлено досрочно");
                isEarlyStopping = true;
                break;
            }
        }
        if (isEarlyStopping) break;
    }

    // Конец обучения
    foreach (var callback in _callbacks)
        callback.OnTrainEnd();

    Console.WriteLine("\n=== Обучение завершено ===");
}
```

**Что здесь происходит?**

### Подготовка

Перед началом выводим информацию о том, что будем учить. Это полезно для отладки и просто для красоты.

### Начало обучения

Вызываем `OnTrainBegin` у всех колбэков. Кто-то может создать файл лога, кто-то — инициализировать таймер.

### Цикл по эпохам

Для каждой эпохи:

1. Уведомляем колбэки о начале эпохи
2. Обучаемся на тренировочных данных
3. Сохраняем историю тренировочных метрик
4. Валидируемся (если есть валидационный датасет)
5. Сохраняем историю валидационных метрик
6. Печатаем прогресс
7. Уведомляем колбэки о конце эпохи
8. Проверяем, не пора ли остановиться (early stopping)

### Завершение

Вызываем `OnTrainEnd` у колбэков. Можно закрыть файлы, отправить уведомление, построить графики.

## TrainingStats: Структура для результатов

```csharp
public struct TrainingStats
{
    public double Loss;      // Потери
    public double Accuracy;  // Доля правильных ответов
}
```

Простая структура, которая группирует две основные метрики. Удобно возвращать из методов одно значение вместо двух.

## Пример использования

Давайте посмотрим, как это работает в реальном коде:

```csharp
// Создаем модель
var model = new Sequential();
model.Add(new DenseLayer(784, 128));
model.Add(new ReLU());
model.Add(new DenseLayer(128, 10));

// Создаем оптимизатор
var optimizer = new Adam(model.Parameters(), learningRate: 0.001);

// Создаем функцию потерь
var loss = new CrossEntropyLoss();

// Загружаем данные
var trainLoader = new MNISTDataLoader("train");
var valLoader = new MNISTDataLoader("test");

var trainDataset = trainLoader.Load();
var valDataset = valLoader.Load();

// Создаем тренера
var trainer = new Trainer(model, optimizer, loss, trainDataset, valDataset);

// Добавляем колбэки
trainer.AddCallback(new ModelCheckpoint(model, "checkpoints"));
trainer.AddCallback(new EarlyStopping(patience: 5));

// Запускаем обучение!
trainer.Fit(epochs: 100);

// После обучения можно посмотреть историю
var finalTrainLoss = trainer.TrainLosses.Last();
var finalValLoss = trainer.ValLosses.Last();
Console.WriteLine($"Финальный train loss: {finalTrainLoss}, val loss: {finalValLoss}");
```

## Почему такая архитектура?

### 1. Разделение ответственности

- Модель знает только как считать forward
- Оптимизатор знает только как обновлять веса
- Loss знает только как считать ошибку
- Dataset знает только как хранить данные
- Callback'и знают только как реагировать на события
- Trainer знает только как координировать

Каждый занимается своим делом. Это упрощает тестирование и модификацию.

### 2. Инверсия управления

Trainer не знает, что делают колбэки. Он просто вызывает их в нужные моменты. Это позволяет добавлять новое поведение без изменения кода Trainer'а.

### 3. История обучения

Сохраняем все метрики в списках. Потом можно построить графики, найти лучшую эпоху, проанализировать переобучение.

### 4. Гибкость

Валидационный датасет опционален. Можно учить и без него. Колбэки тоже опциональны — можно не добавлять ни одного.

## Что дальше?

Наш Trainer — это база. На его основе можно построить много интересного:

1. **Distributed training** — обучение на нескольких GPU
2. **Mixed precision** — для ускорения и экономии памяти
3. **Gradient accumulation** — когда батч не влезает в память
4. **Cross-validation** — автоматическая проверка на разных разбиениях
5. **Hyperparameter search** — подбор гиперпараметров

Но для начала и этого достаточно. У нас есть работающий, понятный, гибкий тренер, который умеет:

- Обучать модель на данных
- Валидироваться на отдельной выборке
- Собирать метрики
- Уведомлять колбэки
- Останавливаться при отсутствии прогресса
- Сохранять историю обучения

## Заключение: Дирижер в действии

Trainer — это дирижер, который превращает набор разрозненных музыкантов (слои, оптимизатор, функцию потерь) в слаженный оркестр. Он задает темп (эпохи), следит за ритмом (батчи), дает знак начинать (прямой проход) и заканчивать (обратный проход).

Без тренера нам пришлось бы писать один и тот же цикл обучения для каждой новой задачи. С тренером мы просто говорим: «Вот модель, вот данные, вот как учить — запускай!» А дальше он все делает сам.
