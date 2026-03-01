---
title: "SharpLeNet. Сохранение и загрузка моделей: Бессмертие для нейросетей"
date: 2026-03-01
description: "SharpLeNet. Сохранение и загрузка моделей: Бессмертие для нейросетей"
tags: ["SharpLeNet"]
categories: ["SharpLeNet"]
series: ["SharpLeNet"]
reading_time: 5
featured: true
---

Друзья, мы научились создавать нейросети, обучать их, следить за прогрессом. Но есть одна проблема: **нейросети смертны**. Выключили программу — и всё, обученные веса исчезли. Придется учить заново. А это часы, дни, а иногда и недели вычислений.

Сегодня мы дадим нашим моделям бессмертие. Научимся сохранять их на диск и загружать обратно. И сделаем это правильно — через интерфейсы, чтобы можно было выбирать разные форматы сохранения.

## Зачем сохранять модели?

1. **Чтобы не учить заново** — обучили один раз, сохранили, используем где угодно
2. **Чтобы делать чекпоинты** — сохраняем лучшие версии во время обучения
3. **Чтобы делиться** — передали файл коллеге, и у него та же модель
4. **Чтобы продолжать обучение** — остановились, загрузили, продолжили

## Интерфейс IModelSaver: Контракт на бессмертие

```csharp
public interface IModelSaver
{
    /// <summary>
    /// Сохраняет модель
    /// </summary>
    void Save(Model model, string identifier);

    /// <summary>
    /// Загружает модель
    /// </summary>
    Model Load(string identifier);

    /// <summary>
    /// Проверяет, существует ли модель
    /// </summary>
    bool Exists(string identifier);
}
```

Это наш контракт. Любой класс, который хочет уметь сохранять модели, должен реализовать эти три метода. Что значит `identifier`? Это может быть:

- Имя файла (для файлового сохранятора)
- Ключ в базе данных (для БД-сохранятора)
- URL (для облачного сохранятора)
- Строка подключения (для чего угодно)

Интерфейс не диктует, как именно сохранять. Он только говорит: «Я умею сохранять, загружать и проверять существование».

## Базовый callback для чекпоинтов

```csharp
public abstract class ModelCheckpointBase : Callback
{
    protected readonly IModelSaver _modelSaver;
    protected readonly string _basePath;
    protected double _bestLoss;

    protected ModelCheckpointBase(IModelSaver modelSaver, string basePath)
    {
        _modelSaver = modelSaver ?? throw new ArgumentNullException(nameof(modelSaver));
        _basePath = basePath;
        _bestLoss = double.MaxValue;
    }
}
```

Этот абстрактный класс задает основу для всех колбэков, связанных с сохранением моделей. У него есть:

- `_modelSaver` — сохранятор, который будет делать грязную работу
- `_basePath` — базовый путь (может быть папкой, префиксом и т.д.)
- `_bestLoss` — лучшее значение loss для отслеживания улучшений

## Конкретный ModelCheckpoint

```csharp
public class ModelCheckpoint : Callback
{
    private readonly Model _model;
    private readonly IModelSaver _modelSaver;
    private readonly string _basePath;
    private double _bestLoss;

    public ModelCheckpoint(Model model, IModelSaver modelSaver, string basePath)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _modelSaver = modelSaver ?? throw new ArgumentNullException(nameof(modelSaver));
        _basePath = basePath;
        _bestLoss = double.MaxValue;
    }

    public override void OnEpochEnd(int epoch, double trainLoss, double valLoss,
                                   double trainAcc, double valAcc)
    {
        if (valLoss < _bestLoss)
        {
            _bestLoss = valLoss;
            string identifier = $"epoch_{epoch + 1}_loss_{valLoss:F6}";

            _modelSaver.Save(_model, identifier);
            Console.WriteLine($"  ✓ Модель сохранена (лучшая val_loss: {valLoss:F6})");
        }
    }
}
```

**Как это работает:**

1. На каждой эпохе после валидации получаем `valLoss`
2. Сравниваем с лучшим известным значением
3. Если улучшилось — сохраняем модель
4. В качестве идентификатора используем строку с номером эпохи и значением loss

**Красота подхода:** `ModelCheckpoint` ничего не знает о том, как именно сохраняется модель. Он просто говорит `_modelSaver.Save(_model, identifier)`. А уж файл это будет, база данных или облако — не его забота.

## Преимущества архитектуры

### 1. Разделение ответственности

- `ModelCheckpoint` решает **когда** сохранять (при улучшении метрики)
- `IModelSaver` решает **как** сохранять (формат, носитель)

### 2. Легкая замена

Хотите вместо файлов сохранять в базу данных? Просто реализуйте `IModelSaver`, работающий с БД, и подставьте его в `ModelCheckpoint`. Всё остальное останется без изменений.

### 3. Тестируемость

Можно создать мок-сохранятор для тестов, который не пишет на диск, а только проверяет вызовы.

### 4. Расширяемость

Хотите сохранять не только лучшую модель, но и каждую эпоху? Создайте `EveryEpochCheckpoint`, наследующий от `ModelCheckpointBase`.

## Примеры использования

```csharp
// Создаем модель
var model = new Sequential();
model.Add(new DenseLayer(784, 128));
model.Add(new ReLU());
model.Add(new DenseLayer(128, 10));

// Создаем сохранятор (например, файловый)
IModelSaver saver = new BinaryModelSaver("./models");

// Создаем колбэк с этим сохранятором
var checkpoint = new ModelCheckpoint(model, saver, "mnist_classifier");

// Добавляем в тренера
trainer.AddCallback(checkpoint);

// Обучаем — модель будет сохраняться при каждом улучшении
trainer.Fit(10);
```

## Чего не хватает?

В следующих статьях мы реализуем конкретные сохраняторы:

1. **BinaryModelSaver** — сохраняет модель в бинарном формате (быстро, компактно, но непонятно для человека)
2. **JsonModelSaver** — сохраняет в JSON (читаемо, но объемно)
3. **DatabaseModelSaver** — в базу данных (для продакшна)

Также можно добавить:

- Сжатие (zip/gzip)
- Шифрование (для чувствительных моделей)
- Версионирование
- Метаданные (дата создания, версия фреймворка, гиперпараметры)

## Заключение: Вечная жизнь для нейросетей

Теперь наши модели могут жить вечно. Мы можем:

- Сохранять их во время обучения (best model checkpoint)
- Загружать для инференса (использовать без обучения)
- Продолжать обучение с того места, где остановились
- Делиться моделями с коллегами
- Деплоить в продакшн

И всё это через единый интерфейс, который позволяет подменять стратегию сохранения без изменения остального кода.

В следующей статье мы реализуем первый конкретный сохранятор — бинарный. А потом разберем JSON и другие форматы. Но основа уже заложена: правильная, гибкая, расширяемая.
