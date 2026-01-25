---
title: "SharpLeNet. Тензор: Autograd"
date: 2026-01-25
description: "SharpLeNet. Реализация автоматического дифференцирования"
tags: ["SharpLeNet"]
categories: ["SharpLeNet"]
series: ["SharpLeNet"]
reading_time: 5
featured: true
---

Добро пожаловать на вторую часть реализации тензора и третью статью всего цикла. Здесь мы реализуем алгоритм автоматического дифференцирования, т.е. вычисления производной на лету.

# Автоматическое дифференцирование. Теория.

## 1. Введение

Автоматическое дифференцирование (Autograd) — это фундаментальная техника вычисления производных, лежащая в основе современных фреймворков глубокого обучения. В отличие от символьного дифференцирования и численных методов, Autograd сочетает преимущества обоих подходов, позволяя эффективно вычислять градиенты произвольных вычислимых функций.

## 2. Математические основы

### 2.1. Цепное правило (Chain Rule)

Пусть имеется композиция функций:

```
y = f(g(h(x))) = f(g(h))
```

Тогда производная по x вычисляется как:

```
dy/dx = (df/dg) * (dg/dh) * (dh/dx)
```

Это правило распространяется на многомерные случаи, где производные становятся матрицами Якоби, а умножение — матричным.

### 2.3. Прямой и обратный режимы

**Прямой режим (Forward mode):**

- Вычисляет производные одновременно с прямым проходом
- Эффективен для функций с малым количеством входов и большим количеством выходов

**Обратный режим (Reverse mode):**

- Сначала выполняется прямой проход для вычисления значений
- Затем обратный проход для вычисления градиентов
- Эффективен для функций с большим количеством входов и малым количеством выходов (типично для нейросетей)

## 3. Граф вычислений

### 3.1. Структура графа

Каждая операция в Autograd представляется как узел в ориентированном ациклическом графе:

```
    x (лист)        w (лист)
      |               |
      v               v
    Умножение --------+
        |
        v
    Сложение <------- b (лист)
        |
        v
        z
```

**Типы узлов:**

- **Листья (Leaves):** Входные тензоры (данные, параметры)
- **Внутренние узлы:** Результаты операций
- **Корень:** Финальный выход (обычно функция потерь)

### 3.2. Атрибуты узлов

Каждый узел содержит:

- Значение тензора
- Функцию производной (grad_fn)
- Ссылки на родительские узлы
- Градиент (изначально None)

## 4. Алгоритм Autograd

### 4.1. Прямой проход (Forward Pass)

1. Инициализация входных тензоров с флагом `requires_grad=True`
2. Выполнение операций с построением графа:

   ```python
   # Пример: z = w*x + b
   x = torch.tensor(2.0, requires_grad=True)
   w = torch.tensor(3.0, requires_grad=True)
   b = torch.tensor(1.0, requires_grad=True)

   # Прямой проход с построением графа
   y = w * x    # Узел умножения
   z = y + b    # Узел сложения
   ```

3. Запись операций в граф вычислений

### 4.2. Обратный проход (Backward Pass)

Алгоритм обратного распространения:

1. **Инициализация:**
   - Устанавливаем градиент выходного узла в 1.0 (dz/dz = 1)
   - Создаем очередь узлов для обработки

2. **Обратное распространение:**
   Для каждого узла в обратном топологическом порядке:

   ```
   Алгоритм backward(node, grad_output):
       # grad_output - градиент от дочернего узла

       # 1. Сохраняем градиент для текущего узла
       if node.grad is None:
           node.grad = grad_output
       else:
           node.grad += grad_output  # Для ветвлений

       # 2. Вычисляем градиенты для родителей
       if node имеет операцию градиента (grad_fn):
           # Получаем локальные градиенты
           local_grads = grad_fn(grad_output)

           # 3. Распространяем на родителей
           for parent, local_grad in zip(node.parents, local_grads):
               backward(parent, local_grad)
   ```

3. **Пример для z = w\*x + b:**
   ```
   Прямой проход:         Обратный проход:
   x = 2                  dz/dz = 1
   w = 3                  dz/dy = 1 (из сложения)
   y = w*x = 6            dz/db = 1
   z = y+b = 7            dz/dw = dz/dy * dy/dw = 1 * x = 2
                          dz/dx = dz/dy * dy/dx = 1 * w = 3
   ```

### 4.3. Вычисление локальных градиентов

Каждая операция определяет свою функцию градиента:

**Умножение:** z = x \* y

```
dz/dx = y
dz/dy = x
```

**Сложение:** z = x + y

```
dz/dx = 1
dz/dy = 1
```

**Сигмоида:** σ(x) = 1/(1+exp(-x))

```
dσ/dx = σ(x) * (1 - σ(x))
```

**Матричное умножение:** C = A @ B

```
dL/dA = dL/dC @ B^T
dL/dB = A^T @ dL/dC
```

## 5. Особенности реализации

### 5.1. Динамический граф (как в PyTorch)

Граф строится на лету во время выполнения:

- Гибкость: позволяет использовать условия и циклы
- Проще для отладки
- Большие накладные расходы на каждую итерацию

### 5.2. Статический граф (как в TensorFlow)

Граф строится заранее, затем выполняется:

- Высокая производительность
- Возможности оптимизации графа
- Сложнее для отладки

## 6. Оптимизации

### 6.1. Отслеживание градиентов

Флаг `requires_grad` позволяет контролировать, для каких тензоров строить граф:

```python
# Только w требует градиенты
x = torch.tensor(2.0)           # requires_grad=False по умолчанию
w = torch.tensor(3.0, requires_grad=True)
```

### 6.2. Контекст no_grad

Отключает построение графа для экономии памяти:

```python
with torch.no_grad():
    # Операции без отслеживания градиентов
    inference_output = model(input_data)
```

### 6.3. Освобождение графа

```python
z.backward()  # Вычисляет градиенты
# Граф автоматически очищается после backward()
```

## 7. Применение в обратном распространении (Backpropagation)

### 7.1. Обучение нейронной сети

```
for epoch in range(epochs):
    # Прямой проход
    output = model(input)
    loss = loss_function(output, target)

    # Обратный проход
    loss.backward()  # Autograd вычисляет все градиенты

    # Обновление параметров
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

        # Обнуление градиентов
        model.zero_grad()
```

### 7.2. Вычисление градиентов для разных узлов

```python
# Вычисление градиентов относительно разных выходов
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = torch.tensor([3.0, 4.0], requires_grad=True)

z1 = x * y
z2 = x + y

# Градиент относительно z1
torch.autograd.backward(z1, grad_tensors=torch.ones_like(z1))
print(x.grad)  # [3., 4.] (∂z1/∂x = y)

# Очистка градиентов
x.grad = None
y.grad = None

# Градиент относительно z2
torch.autograd.backward(z2, grad_tensors=torch.ones_like(z2))
print(x.grad)  # [1., 1.] (∂z2/∂x = 1)
```

## 8. Заключение

Autograd является краеугольным камнем современного глубокого обучения, позволяя исследователям и инженерам сосредоточиться на архитектуре моделей, а не на ручном вычислении градиентов. Понимание его работы необходимо для эффективной отладки, создания новых архитектур и оптимизации производительности нейронных сетей.

**Ключевые преимущества:**

- Автоматизация сложных вычислений производных
- Поддержка динамических вычислительных графов
- Эффективное использование памяти
- Интеграция с аппаратным ускорением (GPU/TPU)

# Автоматическое дифференцирование. Реализация.

Теперь перейдем от сложной теории к простой практике. Для начала, дополним наш класс Tensor следующим полями:

```csharp
/// <summary>
/// Градиент. Тензор той же формы.
/// </summary>
public Tensor? Grad { get; set; }

/// <summary>
/// Левый операнд (родительский тензор)
/// </summary>
public Tensor? LeftParent { get; private set; }

/// <summary>
/// Правый операнд (родительский тензор)
/// </summary>
public Tensor? RightParent { get; private set; }

/// <summary>
/// Операция, породившая тензор
/// </summary>
public TensorOperation Operation { get; private set; }

/// <summary>
/// Указывает, вычислен ли градиент
/// </summary>
public bool RequiresGrad { get; private set; }
```

Далее, дополним наши конструкторы флагом `RequiresGrad`, а также создадим еще один - для указания тензоров, породивших текущий, и операции, благодаря которой был создан этот тензор:

```csharp
public Tensor(double[] data, int[] shape, Tensor? left, Tensor? right,
    TensorOperation operation, bool requiresGrad = false)
{
    if (data.Length != shape.Aggregate(1, (a, b) => a * b))
        throw new ArgumentException("Общее количество элементов должно совпадать с произведением размерностей!");
    Data = data;
    Shape = (int[])shape.Clone();

    ComputeStrides();

    LeftParent = left;
    RightParent = right;
    Operation = operation;
    RequiresGrad = requiresGrad
        || (LeftParent?.RequiresGrad == true)
        || (RightParent?.RequiresGrad == true);

    if (RequiresGrad)
    {
        Grad = new Tensor(shape);
    }
}

public Tensor(double[] data, int[] shape, bool requiresGrad = false)
    : this(data, shape, null, null, TensorOperation.None, requiresGrad)
{
}

public Tensor(int[] shape, bool requiresGrad = false)
    : this(new double[shape.Aggregate(1, (a, b) => a * b)], shape, requiresGrad)
{
}
```

Как можно заметить, для типов операций создан собственный enum:

```csharp
namespace SharpLeNet.Core;

/// <summary>
/// Операция над тензором
/// </summary>
public enum TensorOperation
{
    /// <summary>
    /// Дефолтное значение
    /// </summary>
    None,
    /// <summary>
    /// Добавление
    /// </summary>
    Add,

    /// <summary>
    /// Умножение (поэлементное)
    /// </summary>
    Mul,

    /// <summary>
    /// Матричное умножение
    /// </summary>
    MatMul,

    /// <summary>
    /// Сигомоида
    /// </summary>
    Sigmoid,

    /// <summary>
    /// Операция отрицания
    /// </summary>
    Neg,

    /// <summary>
    /// Операция суммирования всех элементов тензора (в результате - скаляр)
    /// </summary>
    Sum
}
```

Пол дела сделано. Осталось реализовать сами эти операции. Вынесем их в статический класс, сделав методами-расширений, чтобы не загромождать основной класс тензора:

```csharp
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLeNet.Core;

public static class TensorOperations
{
    /// <summary>
    /// Операция сложения тензоров
    /// </summary>
    /// <param name="a">Левый операнд</param>
    /// <param name="b">Правый операнд</param>
    public static Tensor Add(this Tensor a, Tensor b)
    {
        if (!a.Shape.SequenceEqual(b.Shape))
            throw new ArgumentException("Кол-во измерений и их размерности у слагаемых " +
                "должны совпадать!");
        double[] resultData = new double[a.Size];
        for (int i = 0; i <  a.Size; i++)
        {
            resultData[i] = a.Data[i] + b.Data[i];
        }
        return new Tensor(resultData, a.Shape, a, b, TensorOperation.Add,
            a.RequiresGrad || b.RequiresGrad);
    }

    /// <summary>
    /// Операция умножения тензоров (поэлементная)
    /// </summary>
    /// <param name="a">Левый операнд</param>
    /// <param name="b">Правый операнд</param>
    /// <exception cref="ArgumentException"></exception>
    public static Tensor Mul(this Tensor a, Tensor b)
    {
        if (!a.Shape.SequenceEqual(b.Shape))
            throw new ArgumentException("Кол-во измерений и их размерности у множителей " +
                "должны совпадать!");
        double[] resultData = new double[a.Size];
        for (int i = 0; i < a.Size; i++)
        {
            resultData[i] = a.Data[i] * b.Data[i];
        }
        return new Tensor(resultData, a.Shape, a, b,
            TensorOperation.Mul, a.RequiresGrad || b.RequiresGrad);
    }

    /// <summary>
    /// Операция матричного умножения тензоров
    /// </summary>
    /// <param name="a">Левый операнд</param>
    /// <param name="b">Правый операнд</param>
    /// <exception cref="ArgumentException"></exception>
    public static Tensor MatMul(this Tensor a, Tensor b)
    {
        if (a.Rank != 2 || b.Rank != 2)
            throw new ArgumentException("Операция матричного умножения может быть" +
                " выполнена только для матриц!");
        if (a.Shape[1] != b.Shape[0])
            throw new ArgumentException($"Кол-во столбцов первой матрицы должно " +
                $"совпадать с кол-вом строк второй матрицы! Размерности текущих матриц: " +
                $"[{a.Shape[0]}, {a.Shape[1]}] x [{b.Shape[0]}, {b.Shape[1]}]");

        int m = a.Shape[0];
        int n = a.Shape[1];
        int p = b.Shape[1];

        double[] resultData = new double[m * p];
        int[] resultShape = new int[] { m, p };

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < p; j++)
            {
                double sum = 0;
                for (int k = 0; k < n; k++)
                {
                    sum += a[i, k] * b[k, j];
                }
                resultData[i * p + j] = sum;
            }
        }

        return new Tensor(resultData, resultShape, a, b, TensorOperation.MatMul,
            a.RequiresGrad || b.RequiresGrad);
    }

    /// <summary>
    /// Операция вычисления сигмоиды
    /// </summary>
    /// <param name="a">Тензор, для которго выполняется операция</param>
    public static Tensor Sigmoid(this Tensor a)
    {
        var resultData = new double[a.Size];
        for (int i = 0; i < a.Size; i++)
        {
            resultData[i] = 1.0 / (1.0 + Math.Exp(-a.Data[i]));
        }

        return new Tensor(resultData, a.Shape, a, null, TensorOperation.Sigmoid,
            a.RequiresGrad);
    }

    /// <summary>
    /// Операция сложения всех элементов тензора.
    /// В результате - скалярная сумма всех элементов
    /// </summary>
    /// <param name="a">Тензор, для которого выполняется операция</param>
    public static Tensor Sum(this Tensor a)
    {
        double sum = 0;
        for (int i = 0; i < a.Size; i++)
        {
            sum += a.Data[i];
        }

        return new Tensor([sum], [1], a, null,
            TensorOperation.Sum, a.RequiresGrad);
    }

    /// <summary>
    /// Операция отрицания
    /// </summary>
    /// <param name="a">Тензор, для которого выполняется отрицание</param>
    public static Tensor Neg(this Tensor a)
    {
        double[] result = new double[a.Size];
        for (int i = 0; i < a.Size; i++)
        {
            result[i] = -a.Data[i];
        }

        return new Tensor(result, a.Shape, a, null, TensorOperation.Neg,
            a.RequiresGrad);
    }
}

```

Далее, для удобства, перегрузим все возможные операторы:

```csharp
public static Tensor operator +(Tensor a, Tensor b) => a.Add(b); // поэлементное сложение

public static Tensor operator *(Tensor a, Tensor b) => a.Mul(b); // поэлементное умножение

public static Tensor operator -(Tensor a) => a.Neg(); // отрицание

// Для матричного умножения в C#, к сожалению, нет оператора, поэтому будем пользоваться методом
```

Дело осталось за малым - реализуем Backward, как ту самую процедуру вычисления производных на лету. Как и было сказано в теории, производная сложной функции - это произведение производных тех функций, которые ее составляют. Причем, каскадно. Так и поступим:

```csharp
/// <summary>
/// Обратное распространение ошибки
/// </summary>
/// <param name="gradient">Градиент</param>
public void Backward(Tensor? gradient = null)
{
    if (!RequiresGrad)
        return;

    // Если gradient == null и это скаляр (размер = 1), инициализируем как 1.0
    if (gradient == null)
    {
        if (Size != 1)
            throw new InvalidOperationException("Градиент может быть создан " +
                "для скалярного значения");
        Grad!.Fill(1.0);
    }
    else
    {
        // Суммируем градиенты (для случая, когда тензор используется несколько раз)
        AddGradient(gradient);
    }

    BackwardToParents();
}
```

А где же тут вычисление производных? В самом конце, в методе `BackwardToParents`. Суть основного метода `Backward` в том, чтобы просто добавить полученный градиент к градиенту текущего экземпляра, а затем продолжить выполнение для родителей `LeftParent` и `RightParent`.

```csharp
/// <summary>
/// Выполняет добавление значений градиента к текущим значениям
/// в <see cref="Data"/> тензора
/// </summary>
/// <param name="gradient">Градиент для добавления</param>
/// <exception cref="ArgumentException"></exception>
private void AddGradient(Tensor gradient)
{
    if (!Shape.SequenceEqual(gradient.Shape))
        throw new ArgumentException("Кол-во измерений и их размерность градиента " +
            "должна сопадать с кол-вом измерений и их размерностью у данного экземпляра");
    for (int i = 0; i < Size; i++)
    {
        Grad!.Data[i] += gradient.Data[i];
    }
}

/// <summary>
/// В зависимости от операции, вычисляет градиенты для родителей
/// </summary>
private void BackwardToParents()
{
    switch (Operation)
    {
        case TensorOperation.Add:
            // d(L)/dA = d(L)/dC * 1
            // d(L)/dB = d(L)/dC * 1
            LeftParent?.Backward(Grad);
            RightParent?.Backward(Grad);
            break;
        case TensorOperation.Mul:
            // d(L)/dA = d(L)/dC * B
            // d(L)/dB = d(L)/dC * A
            if (LeftParent != null && RightParent != null)
            {
                Tensor gradForLeft = Grad! * RightParent;
                Tensor gradForRight = Grad! * LeftParent;
                LeftParent.Backward(gradForLeft);
                RightParent.Backward(gradForRight);
            }
            break;

        case TensorOperation.MatMul:
            // dL/dA = dL/dC @ B^T
            // dL/dB = A^T @ dL/dC
            if (LeftParent != null && RightParent != null)
            {
                Tensor gradForLeft = Grad!.MatMul(RightParent.Transpose());
                Tensor gradForRight = LeftParent.Transpose().MatMul(Grad!);
                LeftParent.Backward(gradForLeft);
                RightParent.Backward(gradForRight);
            }
            break;

        case TensorOperation.Sigmoid:
            // d(L)/dx = d(L)/dσ * σ'(x)
            // где σ'(x) = σ(x) * (1 - σ(x))
            if (LeftParent != null)
            {
                // Вычисляем σ'(x) = output * (1 - output)
                // где output = this (результат сигмоиды)
                double[] sigmaPrimeData = new double[Size];
                for (int i = 0; i < Size; i++)
                {
                    sigmaPrimeData[i] = Data[i] * (1 - Data[i]);
                }
                Tensor sigmaPrime = new(sigmaPrimeData, Shape);

                // Умножаем градиент на производную
                Tensor gradForParent = Grad! * sigmaPrime;
                LeftParent.Backward(gradForParent);
            }
            break;

        case TensorOperation.Neg:
            // d(L)/dx = d(L)/d(-x) * (-1)
            if (LeftParent != null)
            {
                Tensor negativeGrad = new(Shape);
                negativeGrad.Fill(-1.0);
                Tensor gradForParent = Grad! * negativeGrad;
                LeftParent.Backward(gradForParent);
            }
            break;

        default:
            // Листовой узел (исходные данные) - не имеет родителей
            break;
    }
}
```

Ну вот и все. Автоматическое вычисление производных у нас готово. Осталось реализовать вычисление для нескольких других функций, например, ReLU, но до этого мы дойдем в следующих частях.
