---
title: "SharpLeNet. Механизм функций потерь: Как ошибка становится учителем"
date: 2026-02-08
description: "SharpLeNet. Механизм функций потерь: Как ошибка становится учителем"
tags: ["SharpLeNet"]
categories: ["SharpLeNet"]
series: ["SharpLeNet"]
reading_time: 5
featured: true
---

Приветствую! В прошлой статье мы познакомились с тремя главными «учителями» нейросетей: MSE, BCE и CrossEntropy. Сегодня заглянем под капот и поймем, как именно эти учителя работают в нашей системе. Это будет техническая, но очень важная экскурсия — мы увидим, как абстрактные математические формулы превращаются в реальный код, который умеет не только считать ошибки, но и указывать путь к их исправлению.

# Механизм функций потерь: Как ошибка становится учителем

## Базовый класс: Абстракция потерь как контракт

Начнем с фундамента — абстрактного класса `Loss`. Это как устав школы, который определяет, каким должен быть любой учитель:

```csharp
namespace SharpLeNet.Core.Losses;

/// <summary>
/// Базовый класс функции потерь
/// </summary>
public abstract class Loss
{
    /// <summary>
    /// Вычисляет значение функции
    /// </summary>
    /// <param name="predictions">Предсказанные значения</param>
    /// <param name="targets">Целевые значения</param>
    public abstract Tensor Compute(Tensor predictions, Tensor targets);

    /// <summary>
    /// Вычисляет значение функции и ее производной
    /// </summary>
    /// <param name="predictions">Предсказанные значения</param>
    /// <param name="targets">Целевые значения</param>
    public abstract (Tensor loss, Tensor grad) ComputeWithGrad(Tensor predictions, Tensor targets);
}
```

**Что здесь важно?** Каждая функция потерь в нашей системе обязана уметь две вещи:

1. **Вычислить саму потерю** — сказать, насколько нейросеть ошиблась
2. **Вычислить градиент** — сказать, как именно исправить ошибку

Это минимальный контракт, который гарантирует: какой бы сложной ни была функция потерь, она всегда сможет и оценить работу, и дать рекомендации по улучшению.

## Математический инструментарий: Новые операции с тензорами

Прежде чем реализовывать сложные функции потерь, нам нужно расширить наш математический арсенал. Добавим следующие операции с тензорами:

```csharp
/// <summary>
/// Поэлементное деление тензоров
/// </summary>
public static Tensor Div(this Tensor a, Tensor b)
{
    // Проверка: размерности должны совпадать!
    if (!a.Shape.SequenceEqual(b.Shape))
        throw new ArgumentException("Размерности операндов должны совпадать!");

    double[] resultData = new double[a.Size];
    for (int i = 0; i < a.Size; i++)
    {
        resultData[i] = a.Data[i] / b.Data[i];
    }

    // Ключевой момент: сохраняем связь с родителями для autograd
    return new(resultData, a.Shape, a, b, TensorOperation.Div,
        a.RequiresGrad || b.RequiresGrad);
}
```

```csharp
/// <summary>
/// Умножение тензора на скаляр
/// </summary>
public static Tensor MulScalar(this Tensor a, double scalar)
{
    double[] resultData = new double[a.Size];
    for (int i = 0; i < a.Size; i++)
    {
        resultData[i] = a.Data[i] * scalar;
    }

    // Скаляр тоже превращаем в тензор для autograd
    Tensor scalarTensor = new([scalar], [1]);

    return new(resultData, a.Shape, a, scalarTensor, TensorOperation.MulScalar,
        a.RequiresGrad);
}
```

Перегрузим операторы в Тензоре.

```csharp
public static Tensor operator /(Tensor a, Tensor b) => a.Div(b);
public static Tensor operator /(Tensor a, double b) => a.DivScalar(b);
public static Tensor operator *(Tensor a, double b) => a.MulScalar(b);
public static Tensor operator *(double a, Tensor b) => b.MulScalar(a);
```

Теперь мы можем писать код, который выглядит почти как математические формулы:

```csharp
// Вместо этого:
Tensor result = a.MulScalar(2.0).Div(b).MulScalar(0.5);

// Пишем вот так:
Tensor result = (a * 2.0) / b * 0.5;
```

## Autograd для новых операций: Как градиенты текут назад

Магия происходит не в прямом проходе, а в обратном. Давайте посмотрим, как autograd обрабатывает наши новые операции:

### Для деления: Делим градиенты по правилам

```csharp
case TensorOperation.Div:
    // C = A / B
    // dL/dA = dL/dC * (1/B)   ← градиент для числителя
    // dL/dB = dL/dC * (-A/(B²)) ← градиент для знаменателя

    if (LeftParent != null && RightParent != null)
    {
        // Градиент для A (числитель): умножаем на обратное B
        var oneOverB = new Tensor(RightParent.Shape);
        for (int i = 0; i < RightParent.Size; i++)
        {
            oneOverB.Data[i] = 1.0 / RightParent.Data[i];
        }
        var gradForLeft = Grad! * oneOverB;
        LeftParent.Backward(gradForLeft);

        // Градиент для B (знаменатель): умножаем на -A/B²
        var minusAOverBSquared = new Tensor(LeftParent.Shape);
        for (int i = 0; i < LeftParent.Size; i++)
        {
            double b = RightParent.Data[i];
            minusAOverBSquared.Data[i] = -LeftParent.Data[i] / (b * b);
        }
        var gradForRight = Grad! * minusAOverBSquared;
        RightParent.Backward(gradForRight);
    }
    break;
```

**Разберем на примере:** Допустим, у нас есть операция `C = A / B`, где:

- `A = [4, 9]`
- `B = [2, 3]`
- `C = [2, 3]`

Предположим, пришел градиент `dL/dC = [0.1, 0.2]`. Тогда:

- `dL/dA = [0.1 * (1/2), 0.2 * (1/3)] = [0.05, 0.066]`
- `dL/dB = [0.1 * (-4/4), 0.2 * (-9/9)] = [-0.1, -0.2]`

**Интуиция:** Если мы делим на большое число B, то небольшие изменения в A сильно меняют результат → градиент для A должен быть ослаблен (умножаем на 1/B). А изменения в B наоборот сильно влияют на результат → градиент для B усиливается (умножаем на A/B²).

### Для умножения на скаляр: Просто, но важно

```csharp
case TensorOperation.MulScalar:
    // C = A * scalar
    // dL/dA = dL/dC * scalar

    if (LeftParent != null && RightParent != null && RightParent.Size == 1)
    {
        double scalar = RightParent.Data[0];

        // Создаем тензор со скаляром той же формы что и градиент
        var scalarTensor = new Tensor(Grad!.Shape);
        scalarTensor.Fill(scalar);

        var gradForParent = Grad! * scalarTensor;
        LeftParent.Backward(gradForParent);
    }
    break;
```

**Почему так просто?** Потому что производная от `C = A * k` — это просто `k`. Если мы умножаем на 2, то градиент тоже умножается на 2. Если делим на 10 (умножаем на 0.1), то градиент уменьшается в 10 раз.

Это критически важно для нормализации! Когда мы вычисляем среднее (делим сумму на N), мы фактически умножаем каждый элемент на 1/N. И градиенты тоже умножаются на 1/N, что предотвращает их взрывное growth.

## Итог

Мы построили мощную инфраструктуру:

- Абстрактный класс Loss задает контракт
- Новые тензорные операции дают математический инструментарий
- Autograd с расширенной логикой обеспечивает обратное распространение
- Перегруженные операторы делают код читаемым

Эта инфраструктура позволяет нам реализовывать любые функции потерь — от простой MSE до сложных перцептуальных loss'ов в GAN'ах — единообразно, эффективно и надежно.

В следующих статьях мы реализуем конкретные функции потерь и увидим, как вся эта теория работает на практике.
