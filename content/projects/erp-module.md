---
title: "Модернизация ERP системы: от Legacy к Современной Архитектуре"
date: 2026-01-10
description: "Полный кейс рефакторинга и архитектурной переработки ERP-системы с внедрением современных .NET практик в .NET Framework проект"
tags:
  [
    ".NET Framework",
    "Архитектура",
    "Legacy",
    "Рефакторинг",
    "DDD",
    "DI",
    "Mediator",
  ]
categories: ["Проекты"]
featured: true
github: "https://github.com/DanilRukin/legacy-modernization-examples"
demo: "/experiments/di-container-demo"
reading_time: 12
---

# Модернизация ERP системы: Практический кейс

## 📊 Контекст проекта

**Проект:** ERP система для энергетической компании с 10-летней историей разработки.

**Исходное состояние:**

- 500K+ строк кода на .NET Framework 4.5.2
- Монолитная архитектура без четкого разделения ответственности
- Высокий технический долг (копипаст, нарушение SRP, God objects)
- Отсутствие unit тестов (менее 5% coverage)
- Сложность добавления новых функций
- Высокий порог входа для новых разработчиков

**Бизнес-проблемы:**

- Медленная разработка новых функций (3+ месяца на среднюю фичу)
- Частые regressions при изменениях
- Высокая стоимость поддержки
- Невозможность масштабирования команды

## 🎯 Цели модернизации

1. **Улучшить поддерживаемость** кодовой базы
2. **Ускорить разработку** новых функций
3. **Повысить качество** кода и уменьшить количество багов
4. **Сделать систему тестируемой**
5. **Снизить порог входа** для новых разработчиков

## 🏗 Архитектурный подход

### **Анализ текущего состояния**

Провели архитектурный аудит и выявили ключевые проблемы:

```csharp
// Типичный код до рефакторинга
public class DataManager
{
    // Нарушение SRP: 2000+ строк кода, отвечает за всё
    public void ProcessEverything()
    {
        // Бизнес-логика
        // Работа с БД
        // Валидация
        // Логирование
        // Отправка email
        // И многое другое...
    }
}
```

## Выбранная стратегия:

- Постепенная миграция (не big bang rewrite)
- Стратегия Strangler Fig Pattern
- Внедрение через новые функции
- Рефакторинг при касании legacy кода

🔧 **Ключевые реализации**

### 1. Собственный DI контейнер для .NET Framework

Проблема: .NET Framework не имеет встроенного DI, а внедрение сторонних решений было невозможно из-за политик безопасности.

Решение: Разработал lightweight IoC контейнер:

```csharp
public class ServiceContainer : IServiceProvider
{
    private readonly Dictionary<Type, ServiceDescriptor> _descriptors = new();

    public void Register<TService, TImplementation>(Lifecycle lifecycle = Lifecycle.Transient)
        where TImplementation : TService
    {
        var descriptor = new ServiceDescriptor(
            typeof(TService),
            typeof(TImplementation),
            lifecycle);

        _descriptors[typeof(TService)] = descriptor;
    }

    public object GetService(Type serviceType)
    {
        if (!_descriptors.ContainsKey(serviceType))
        {
            // Автоматическая регистрация по соглашению
            AutoRegister(serviceType);
        }

        return CreateInstance(_descriptors[serviceType]);
    }

    private object CreateInstance(ServiceDescriptor descriptor)
    {
        // Constructor injection
        var constructor = descriptor.ImplementationType
            .GetConstructors()
            .First();

        var parameters = constructor.GetParameters();
        var dependencies = parameters
            .Select(p => GetService(p.ParameterType))
            .ToArray();

        return constructor.Invoke(dependencies);
    }
}

// Регистрация сервисов
container.Register<ILogger, FileLogger>(Lifecycle.Singleton);
container.Register<IEmailService, SmtpEmailService>();
container.Register<IPaymentProcessor, PaymentProcessor>();

// Разрешение зависимостей
var processor = container.GetService<IPaymentProcessor>();
```

**Особенности реализации:**

- Поддержка Singleton, Scoped, Transient жизненных циклов
- Constructor injection
- Circular dependency detection
- Lazy initialization
- Disposable pattern support

### 2. Middleware слой для cross-cutting concerns

**Проблема:** Повторяющийся код для логирования, обработки ошибок, валидации.

**Решение:** Pipeline-based middleware система:

```csharp
public interface IMiddleware
{
    Task InvokeAsync(HttpContext context, Func<Task> next);
}

public class MiddlewarePipeline
{
    private readonly List<IMiddleware> _middlewares = new();

    public MiddlewarePipeline Use(IMiddleware middleware)
    {
        _middlewares.Add(middleware);
        return this;
    }

    public async Task ExecuteAsync(HttpContext context)
    {
        await ExecuteMiddlewareAsync(context, 0);
    }

    private async Task ExecuteMiddlewareAsync(HttpContext context, int index)
    {
        if (index >= _middlewares.Count)
            return;

        var middleware = _middlewares[index];
        await middleware.InvokeAsync(context,
            () => ExecuteMiddlewareAsync(context, index + 1));
    }
}

// Конкретные middleware
public class LoggingMiddleware : IMiddleware
{
    private readonly ILogger _logger;

    public LoggingMiddleware(ILogger logger)
    {
        _logger = logger;
    }

    public async Task InvokeAsync(HttpContext context, Func<Task> next)
    {
        var sw = Stopwatch.StartNew();
        _logger.LogInformation($"Request: {context.Request.Path}");

        try
        {
            await next();
        }
        finally
        {
            sw.Stop();
            _logger.LogInformation($"Response: {context.Response.StatusCode} - {sw.ElapsedMilliseconds}ms");
        }
    }
}

public class ExceptionHandlingMiddleware : IMiddleware
{
    public async Task InvokeAsync(HttpContext context, Func<Task> next)
    {
        try
        {
            await next();
        }
        catch (ValidationException ex)
        {
            context.Response.StatusCode = 400;
            await context.Response.WriteJsonAsync(new { error = ex.Message });
        }
        catch (Exception ex)
        {
            context.Response.StatusCode = 500;
            await context.Response.WriteJsonAsync(new { error = "Internal server error" });
        }
    }
}

// Использование
var pipeline = new MiddlewarePipeline()
    .Use(new LoggingMiddleware(logger))
    .Use(new ExceptionHandlingMiddleware())
    .Use(new AuthenticationMiddleware())
    .Use(new AuthorizationMiddleware())
    .Use(new RoutingMiddleware(controllers));
```

### 3. Mediator паттерн для декомпозиции

**Проблема:** Tight coupling между контроллерами и бизнес-логикой.

**Решение:** Реализация CQRS-like mediator:

```csharp
public interface IMediator
{
    Task<TResponse> Send<TResponse>(IRequest<TResponse> request);
    Task Publish<TNotification>(TNotification notification)
        where TNotification : INotification;
}

public interface IRequest<TResponse> { }

public interface INotification { }

// Команда
public class CreateInvoiceCommand : IRequest<InvoiceCreatedResponse>
{
    public int CustomerId { get; set; }
    public decimal Amount { get; set; }
    public DateTime DueDate { get; set; }
}

// Обработчик
public class CreateInvoiceHandler : IRequestHandler<CreateInvoiceCommand, InvoiceCreatedResponse>
{
    private readonly IInvoiceRepository _repository;
    private readonly IEmailService _emailService;

    public CreateInvoiceHandler(IInvoiceRepository repository, IEmailService emailService)
    {
        _repository = repository;
        _emailService = emailService;
    }

    public async Task<InvoiceCreatedResponse> Handle(CreateInvoiceCommand request)
    {
        // Бизнес-логика
        var invoice = new Invoice
        {
            CustomerId = request.CustomerId,
            Amount = request.Amount,
            DueDate = request.DueDate,
            Status = InvoiceStatus.Pending
        };

        await _repository.AddAsync(invoice);

        // Domain event
        await _mediator.Publish(new InvoiceCreatedEvent(invoice.Id));

        return new InvoiceCreatedResponse { InvoiceId = invoice.Id };
    }
}

// Контроллер
public class InvoiceController
{
    private readonly IMediator _mediator;

    public InvoiceController(IMediator mediator)
    {
        _mediator = mediator;
    }

    [HttpPost]
    public async Task<IActionResult> Create([FromBody] CreateInvoiceCommand command)
    {
        var result = await _mediator.Send(command);
        return Ok(result);
    }
}
```

## Качественные улучшения

- Архитектурная ясность - понятная структура проекта
- Тестируемость - изолированная бизнес-логика
- Расширяемость - легко добавлять новые функции
- Поддерживаемость - меньше кода, больше ясности
- Командная продуктивность - стандартизированные подходы

## 🎓 Извлеченные уроки

Что сработало хорошо:

- ✅ Постепенная миграция - низкий риск, быстрая отдача
- ✅ Обучение команды - воркшопы, парное программирование
- ✅ Автоматические тесты - confidence для рефакторинга
- ✅ Документация - для каждого нового паттерна

## Вызовы и решения:

- Сопротивление изменениям → Показал быстрые wins на маленьких задачах
- Сложность тестирования legacy → Начинал с integration tests
- Временные затраты → Выделили 20% времени на рефакторинг
- Технический долг → Ввели "технические спринты"

## 🔮 Будущие улучшения

- План на следующие 6 месяцев:
- Миграция на .NET 6 - поэтапный переход
- Внедрение Event Sourcing для критичных бизнес-процессов
- Разделение на микросервисы для scaling
- UI modernization - переход с WPF на Blazor

# 🏆 Ключевые выводы

- Legacy не значит плохо - это возможность для улучшений
- Архитектура важнее технологий - хорошие принципы работают везде
- Команда - ключ к успеху - изменения должны поддерживаться всеми
- Инкрементальные улучшения лучше революций
- Качество кода = скорость разработки - инвестиция, которая окупается
