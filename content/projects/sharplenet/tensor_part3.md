---
title: "SharpLeNet. Тензор: Autograd. Добавление дополнительных операций"
date: 2026-01-26
description: "SharpLeNet. Реализация автоматического дифференцирования - добавление дополнительных операций"
tags: ["SharpLeNet"]
categories: ["SharpLeNet"]
series: ["SharpLeNet"]
reading_time: 5
featured: true
---

Добро пожаловать на третью часть реализации тензора и четвертую статью всего цикла. Здесь мы дополним алгоритм автоматического дифференцирования операциями ReLU, Softmax, Log.

# 1. ReLU (Rectified Linear Unit) - функция активации

ReLU - это простая математическая функция:

```
f(x) = max(0, x)
```

То есть, если x положительный - оставляем как есть, если отрицательный - заменяем на ноль.

Эта функция имеет ряд преимуществ:

1. **Решает проблему "затухающих градиентов"** - градиенты не становятся слишком маленькими
2. **Быстро вычисляется** - проще, чем sigmoid или tanh
3. **Обеспечивает разреженность** - многие нейроны становятся нулевыми
4. **Помогает моделям быстрее обучаться**

Так давайте и реализуем эту простую функцию в виде операции (класс `TensorOperations`):

```csharp
/// <summary>
/// Операция вычисления функции ReLU для тензора.
/// ReLU: max(0, x)
/// </summary>
/// <param name="a">Тензор, для которого выполняется вычисление ReLU</param>
public static Tensor ReLU(this Tensor a)
{
    double[] resultData = new double[a.Size];
    for (int i = 0; i < a.Size; i++)
    {
        resultData[i] = Math.Max(0, a.Data[i]);
    }

    return new Tensor(resultData, a.Shape, a, null, TensorOperation.ReLU,
        a.RequiresGrad);
}
```

Теперь дополним метод `BackwardToParents` класса `Tensor` блоком вычисления производной ReLU. Производная вычисляется как

```
f(x) = max(0, x)

Производная:
f'(x) = 1, если x > 0
f'(x) = 0, если x ≤ 0
```

Ну так давайте так и напишем:

```csharp
case TensorOperation.ReLU:
    // d(L)/dx = d(L)/dReLU * (x > 0 ? 1 : 0)
    if (LeftParent != null)
    {
        double[] reluPrimeData = new double[Size];
        for (int i = 0; i < Size; i++)
        {
            reluPrimeData[i] = LeftParent.Data[i] > 0 ? 1.0 : 0.0;
        }
        Tensor reluPrime = new(reluPrimeData, Shape);
        Tensor gradForParent = Grad! * reluPrime;
        LeftParent.Backward(gradForParent);
    }
    break;
```

# 2. Softmax - функция нормализации

Softmax преобразует набор чисел в вероятности:

```
softmax(x_i) = exp(x_i) / Σ exp(x_j)
```

Все выходы получаются между 0 и 1, и их сумма равна 1.

Эта функция также имеет ряд своих достоинств:

1. **Превращает выходы в вероятности** - удобно для классификации
2. **Подчеркивает максимумы** - делает большие значения еще больше
3. **Используется в последнем слое** для задач классификации

Реализация здесь будет посложнее. Вот такая:

```csharp
/// <summary>
/// Операция вычисления Softmax
/// </summary>
/// <param name="a">Тензор, для которого выполняется вычисление Softmax</param>
public static Tensor Softmax(this Tensor a)
{
    if (a.Rank != 2)
        throw new NotImplementedException("Softmax поддерживается только для матриц!");

    int batchSize = a.Shape[0];
    int numClasses = a.Shape[1];

    var resultData = new double[a.Size];

    for (int i = 0; i < batchSize; i++)
    {
        // Находим максимум для численной стабильности
        double maxVal = double.MinValue;
        for (int j = 0; j < numClasses; j++)
        {
            if (a[i, j] > maxVal) maxVal = a[i, j];
        }

        // Вычисляем экспоненты
        double sumExp = 0;
        double[] exps = new double[numClasses];
        for (int j = 0; j < numClasses; j++)
        {
            exps[j] = Math.Exp(a[i, j] - maxVal);
            sumExp += exps[j];
        }

        // Нормализуем
        for (int j = 0; j < numClasses; j++)
        {
            resultData[i * numClasses + j] = exps[j] / sumExp;
        }
    }

    return new Tensor(resultData, a.Shape, a, null, TensorOperation.Softmax,
        a.RequiresGrad);
}
```

Ну и производная для `Softmax` следующая:

```
Если i = j: ∂softmax_i/∂x_j = softmax_i * (1 - softmax_i)
Если i ≠ j: ∂softmax_i/∂x_j = -softmax_i * softmax_j
```

Давайте и этот блок реализуем в классе `Tensor`:

```csharp
case TensorOperation.Softmax:
    // Производная softmax сложная, но мы ее реализуем через CrossEntropy позже
    // Пока оставим placeholder
    throw new NotImplementedException("Вычисление производной Softmax будет " +
        "реализовано позже через CrossEntropy");
    break;
```

Это достаточно сложное вычисление производной, поэтому, очень часто Softmax используют в паре с CrossEntropy. Вот так:

```csharp
/// <summary>
/// Вычисляет Softmax + CrossEntropy Loss (вместе для эффективности)
/// </summary>
/// <param name="logits"></param>
/// <param name="labels"></param>
/// <exception cref="ArgumentException"></exception>
public static (Tensor softmaxOutput, Tensor loss) SoftmaxCrossEntropy(
    this Tensor logits, Tensor labels)
{
    if (logits.Rank != 2 || labels.Rank != 2)
        throw new ArgumentException("Оба тензора должны быть матрицами!");
    if (!logits.Shape.SequenceEqual(labels.Shape))
        throw new ArgumentException("Измерения и их размерности должны совпадать!");

    int batchSize = logits.Shape[0];
    int numClasses = logits.Shape[1];

    // Вычисляем Softmax
    Tensor softmaxOutput = logits.Softmax();

    // Вычисляем Cross-Entropy Loss
    double lossValue = 0;
    for (int i = 0; i < batchSize; i++)
    {
        for (int j = 0; j < numClasses; j++)
        {
            // L = -Σ y_true * log(y_pred)
            // где y_pred = softmax_output
            lossValue -= labels[i, j] * Math.Log(softmaxOutput[i, j] + 1e-10); // добавляем epsilon для стабильности
        }
    }
    lossValue /= batchSize; // // усредняем по батчу

    Tensor loss = new([lossValue], [1], logits, labels, TensorOperation.SoftmaxCrossEntropy,
        logits.RequiresGrad || labels.RequiresGrad);

    return (softmaxOutput, loss);
}
```

Производная в таком случае получается невероятно простой:

```csharp
case TensorOperation.SoftmaxCrossEntropy:
    // Магически простой градиент: dL/dz = softmax(z) - y_true
    if (LeftParent != null && RightParent != null) // LeftParent = logits, RightParent = labels
    {
        int batchSize = LeftParent.Shape[0];
        int numClasses = LeftParent.Shape[1];

        // Вычисляем softmax(z) - y
        Tensor gradForParent = new Tensor(LeftParent.Shape);

        // Сначала вычисляем softmax
        Tensor softmax = LeftParent.Softmax();

        // Вычисляем градиент: softmax - labels
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < numClasses; j++)
            {
                gradForParent[i, j] = (softmax[i, j] - RightParent[i, j]) / batchSize;
            }
        }
        // Передаем градиент только к logits (labels не обучаются)
        LeftParent.Backward(gradForParent);
    }
    break;

```

# 3. Log (Логарифм) - для вычисления потерь

Log - это математическая функция, обратная к экспоненте. В нейросетях обычно используется натуральный логарифм (ln).
Тоже имеет ряд преимуществ:

1. **Преобразует умножение в сложение** - упрощает вычисления
2. **Уменьшает численные проблемы** - большие числа становятся управляемыми
3. **Используется в функции потерь** - особенно с Softmax

## Комбинация Softmax + Log

Почему их используют вместе?

**Проблема:** Softmax может давать очень маленькие или очень большие числа
**Решение:** Используем log(softmax(...)), что называется LogSoftmax

Фактически вместо:

```
ошибка = -log(softmax(правильный_класс))
```

Многие фреймворки вычисляют это эффективнее как:

```
LogSoftmax(x) = log(softmax(x))
```

Реализуем операцию вычисления логарифма следующим образом:

```csharp
/// <summary>
/// Операция вычисления логарифма
/// </summary>
/// <param name="a">Тензор, для которого выполняется вычисление логарифма</param>
public static Tensor Log(this Tensor a)
{
    double[] resultData = new double[a.Size];
    for (int i = 0; i < a.Size; i++)
    {
        resultData[i] = Math.Log(a.Data[i]);
    }

    return new Tensor(resultData, a.Shape, a, null, TensorOperation.Log,
        a.RequiresGrad);
}
```

У логарифма также есть своя производная. Выглядит она так:

```
f(x) = log(x)
f'(x) = 1/x
```

Ровно это давайте и напишем в классе `Tensor`:

```csharp
case TensorOperation.Log:
    // d(L)/dx = d(L)/d(log(x)) * (1/x)
    if (LeftParent != null)
    {
        double[] logPrimeData = new double[Size];
        for (int i = 0; i < Size; i++)
        {
            logPrimeData[i] = 1.0 / LeftParent.Data[i];
        }
        Tensor logPrime = new(logPrimeData, Shape);
        Tensor gradForParent = Grad! * logPrime;
        LeftParent.Backward(gradForParent);
    }
    break;
```

# 6. Итог простыми словами

- **ReLU** - "выпрямитель", который делает отрицательные числа нулями
- **Softmax** - "нормализатор", превращает числа в вероятности
- **Log** - "упроститель", помогает работать с очень большими/малыми числами
- **Autograd** - "автоматический калькулятор производных", который соединяет все это вместе

Вместе они позволяют нейросетям обучаться на ошибках и становиться умнее!
