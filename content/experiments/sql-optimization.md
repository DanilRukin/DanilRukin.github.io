---
title: "SQL Оптимизация: от 30 минут до 1 минуты - реальный кейс"
date: 2026-01-10
description: "Интерактивное демо оптимизации SQL запросов с реальными примерами из production. Показываю как ускорить запросы в 30 раз."
tags:
  [
    "SQL",
    "Оптимизация",
    "Performance",
    "MS SQL Server",
    "Database",
    "Query Tuning",
  ]
categories: ["Эксперименты"]
type: "demo"
layout: "demo"
featured: true
github: "https://github.com/DanilRukin/sql-optimization-examples"
reading_time: 8
---

# SQL Оптимизация: Практическое руководство

## 🔍 Контекст проблемы

**Проект:** ERP система для учета энергоресурсов  
**Проблема:** Ежедневный отчет по начислениям выполнялся **30+ минут**  
**Бизнес-воздействие:** Операторы ждали отчеты, задерживались выплаты  
**Данные:** 1.5 млн клиентов, 10 млн платежей, 5 млн счетов

## 📊 Исходный запрос (медленный)

```sql
-- Время выполнения: 30+ минут
-- CPU: 95%
-- Logical Reads: 15,000,000

SELECT
    c.id as CustomerId,
    c.full_name as CustomerName,
    c.account_number,
    SUM(p.amount) as TotalPaid,
    COUNT(p.id) as PaymentCount,
    AVG(p.amount) as AveragePayment,
    MAX(p.payment_date) as LastPaymentDate,
    SUM(CASE WHEN p.status = 'overdue' THEN p.amount ELSE 0 END) as OverdueAmount,
    (SELECT COUNT(*) FROM invoices i
     WHERE i.customer_id = c.id
     AND i.status = 'pending') as PendingInvoicesCount
FROM customers c
LEFT JOIN payments p ON c.id = p.customer_id
LEFT JOIN invoices i ON p.invoice_id = i.id
WHERE c.is_active = 1
    AND c.registration_date >= '2020-01-01'
    AND p.payment_date BETWEEN '2023-01-01' AND '2023-12-31'
    AND p.status IN ('completed', 'overdue')
GROUP BY c.id, c.full_name, c.account_number
HAVING SUM(p.amount) > 0
ORDER BY TotalPaid DESC
OFFSET 0 ROWS
FETCH NEXT 100 ROWS ONLY;
```

## 🕵️‍♂️ Анализ проблем

**Execution Plan показал:**

- Table Scans вместо Index Seeks
- Key Lookups (5 млн операций)
- Hash Joins на больших таблицах
- Скалярные подзапросы в SELECT (N+1 проблема)
- Отсутствие покрывающих индексов

```
Table 'customers'. Scan count 1, logical reads 150,000
Table 'payments'. Scan count 1, logical reads 8,500,000
Table 'invoices'. Scan count 1, logical reads 6,350,000
Total logical reads: 15,000,000
```

## 🛠 Пошаговая оптимизация

**Шаг 1: Анализ и создание индексов**

```sql
-- Выносим фильтрацию во временные таблицы
CREATE TABLE #FilteredCustomers (
    id INT PRIMARY KEY,
    full_name NVARCHAR(200),
    account_number VARCHAR(50),
    INDEX IX_Account (account_number)
);

INSERT INTO #FilteredCustomers (id, full_name, account_number)
SELECT id, full_name, account_number
FROM customers
WHERE is_active = 1
    AND registration_date >= '2020-01-01';

-- Аналогично для payments
CREATE TABLE #FilteredPayments (
    id INT,
    customer_id INT,
    amount DECIMAL(18,2),
    payment_date DATE,
    status VARCHAR(20),
    invoice_id INT,
    INDEX IX_Customer (customer_id),
    INDEX IX_Date (payment_date)
);

INSERT INTO #FilteredPayments
SELECT id, customer_id, amount, payment_date, status, invoice_id
FROM payments
WHERE payment_date BETWEEN '2023-01-01' AND '2023-12-31'
    AND status IN ('completed', 'overdue');

```

**Шаг 2: Временные таблицы для фильтрации**

```sql
-- Выносим фильтрацию во временные таблицы
CREATE TABLE #FilteredCustomers (
    id INT PRIMARY KEY,
    full_name NVARCHAR(200),
    account_number VARCHAR(50),
    INDEX IX_Account (account_number)
);

INSERT INTO #FilteredCustomers (id, full_name, account_number)
SELECT id, full_name, account_number
FROM customers
WHERE is_active = 1
    AND registration_date >= '2020-01-01';

-- Аналогично для payments
CREATE TABLE #FilteredPayments (
    id INT,
    customer_id INT,
    amount DECIMAL(18,2),
    payment_date DATE,
    status VARCHAR(20),
    invoice_id INT,
    INDEX IX_Customer (customer_id),
    INDEX IX_Date (payment_date)
);

INSERT INTO #FilteredPayments
SELECT id, customer_id, amount, payment_date, status, invoice_id
FROM payments
WHERE payment_date BETWEEN '2023-01-01' AND '2023-12-31'
    AND status IN ('completed', 'overdue');
```

**Шаг 3: Устранение N+1 подзапросов**

```sql
-- Вместо скалярного подзапроса используем JOIN
CREATE TABLE #PendingInvoices (
    customer_id INT PRIMARY KEY,
    count INT
);

INSERT INTO #PendingInvoices (customer_id, count)
SELECT customer_id, COUNT(*)
FROM invoices
WHERE status = 'pending'
GROUP BY customer_id;
```

**Шаг 4: Оптимизированный запрос**

```sql
-- Время выполнения: 1 минута
-- CPU: 25%
-- Logical Reads: 150,000 (100x improvement)

WITH CustomerPayments AS (
    SELECT
        fp.customer_id,
        SUM(fp.amount) as TotalPaid,
        COUNT(fp.id) as PaymentCount,
        AVG(fp.amount) as AveragePayment,
        MAX(fp.payment_date) as LastPaymentDate,
        SUM(CASE WHEN fp.status = 'overdue' THEN fp.amount ELSE 0 END) as OverdueAmount
    FROM #FilteredPayments fp
    GROUP BY fp.customer_id
    HAVING SUM(fp.amount) > 0
)
SELECT
    fc.id as CustomerId,
    fc.full_name as CustomerName,
    fc.account_number,
    cp.TotalPaid,
    cp.PaymentCount,
    cp.AveragePayment,
    cp.LastPaymentDate,
    cp.OverdueAmount,
    ISNULL(pi.count, 0) as PendingInvoicesCount
FROM #FilteredCustomers fc
INNER JOIN CustomerPayments cp ON fc.id = cp.customer_id
LEFT JOIN #PendingInvoices pi ON fc.id = pi.customer_id
ORDER BY cp.TotalPaid DESC
OFFSET 0 ROWS
FETCH NEXT 100 ROWS ONLY;

-- Очистка временных таблиц
DROP TABLE #FilteredCustomers;
DROP TABLE #FilteredPayments;
DROP TABLE #PendingInvoices;
```

# 🔧 Инструменты для оптимизации

## 1. SQL Server Profiler / Extended Events

```sql
-- Настройка Extended Events для отслеживания медленных запросов
CREATE EVENT SESSION [SlowQueries] ON SERVER
ADD EVENT sqlserver.sql_statement_completed(
    ACTION(sqlserver.sql_text, sqlserver.plan_handle)
    WHERE ([duration] > 30000000)) -- > 30 секунд
ADD TARGET package0.ring_buffer
WITH (MAX_MEMORY=4096 KB, EVENT_RETENTION_MODE=ALLOW_SINGLE_EVENT_LOSS);
```

## 2. Execution Plan анализ

```sql
-- Включение статистики
SET STATISTICS IO ON;
SET STATISTICS TIME ON;
SET STATISTICS PROFILE ON;

-- Анализ плана выполнения
SELECT *
FROM sys.dm_exec_query_stats AS qs
CROSS APPLY sys.dm_exec_query_plan(qs.plan_handle) AS qp
WHERE qp.objectid = OBJECT_ID('YourProcedureName');
```

## 3. Missing Index Recommendations

```sql
-- Поиск недостающих индексов
SELECT
    migs.avg_total_user_cost * (migs.avg_user_impact / 100.0) * (migs.user_seeks + migs.user_scans) AS improvement_measure,
    'CREATE INDEX [IX_' + OBJECT_NAME(mid.object_id) + '_' + REPLACE(REPLACE(REPLACE(
        ISNULL(mid.equality_columns,'') + ISNULL(mid.inequality_columns,''), ', ', '_'),
        '[', ''), ']', '') + ']' +
    ' ON ' + mid.statement + ' (' +
        ISNULL(mid.equality_columns,'') +
        CASE WHEN mid.equality_columns IS NOT NULL AND mid.inequality_columns IS NOT NULL THEN ',' ELSE '' END +
        ISNULL(mid.inequality_columns, '') + ')' +
    ISNULL(' INCLUDE (' + mid.included_columns + ')', '') AS create_index_statement
FROM sys.dm_db_missing_index_group_stats AS migs
INNER JOIN sys.dm_db_missing_index_groups AS mig ON migs.group_handle = mig.index_group_handle
INNER JOIN sys.dm_db_missing_index_details AS mid ON mig.index_handle = mid.index_handle
WHERE migs.avg_total_user_cost * (migs.avg_user_impact / 100.0) * (migs.user_seeks + migs.user_scans) > 10
ORDER BY improvement_measure DESC;
```

# 📚 Паттерны оптимизации

## 1. Покрывающие индексы (Covering Indexes)

```sql
-- Плохо: Index Seek + Key Lookup
CREATE INDEX IX_Customers_City ON customers(city);

-- Хорошо: Covering Index
CREATE INDEX IX_Customers_City
ON customers(city)
INCLUDE (name, email, phone);
```

## 2. Batch операции вместо RBAR (Row By Agonizing Row)

```sql
-- Плохо: RBAR
DECLARE @id INT = 1;
WHILE @id <= 100000
BEGIN
    UPDATE customers SET status = 'active' WHERE id = @id;
    SET @id += 1;
END

-- Хорошо: Batch
UPDATE customers
SET status = 'active'
WHERE id BETWEEN 1 AND 100000;
```

## 3. Оптимизация JOIN порядков

```sql
-- Плохо: Большая таблица первой
SELECT *
FROM large_table l
JOIN small_table s ON l.id = s.large_id;

-- Хорошо: Маленькая таблица первой
SELECT *
FROM small_table s
JOIN large_table l ON s.large_id = l.id;
```

## 4. Использование EXISTS вместо IN

```sql
-- Плохо: IN с подзапросом
SELECT * FROM orders
WHERE customer_id IN (SELECT id FROM customers WHERE status = 'active');

-- Хорошо: EXISTS
SELECT * FROM orders o
WHERE EXISTS (SELECT 1 FROM customers c
              WHERE c.id = o.customer_id AND c.status = 'active');
```

# 🚫 Анти-паттерны

## 1. SELECT \*

```sql
-- Плохо
SELECT * FROM customers;

-- Хорошо
SELECT id, name, email FROM customers;
```

## 2. Functions в WHERE clause

```sql
-- Плохо (не использует индекс)
SELECT * FROM orders
WHERE YEAR(order_date) = 2023;

-- Хорошо (использует индекс)
SELECT * FROM orders
WHERE order_date >= '2023-01-01'
  AND order_date < '2024-01-01';
```

## 3. Неправильные типы данных в JOIN

```sql
-- Плохо (implicit conversion)
SELECT * FROM customers c
JOIN orders o ON c.id = o.customer_id
WHERE c.account_number = 12345; -- account_number VARCHAR

-- Хорошо
SELECT * FROM customers c
JOIN orders o ON c.id = o.customer_id
WHERE c.account_number = '12345';
```

# 🎯 Практические советы

## Ежедневный чеклист:

- Мониторь медленные запросы через DMV
- Проверяй статистику индексов (fragmentation, usage)
- Анализируй execution plans после изменений
- Тестируй на realistic данных (не на маленькой БД)
- Документируй оптимизации и их impact

## Когда вызывать DBA:

- Запросы > 10 секунд в production
- Блокировки (deadlocks)
- Рост tempdb
- Проблемы с memory pressure
- Need для partitioning

# 🏆 Ключевые выводы

## Что сработало:

- ✅ Covering indexes - устранили key lookups
- ✅ Temporary tables - предварительная фильтрация
- ✅ Batch operations - вместо row-by-row
- ✅ Proper JOIN order - маленькие таблицы first
- ✅ Eliminate N+1 - скалярные подзапросы → JOIN

## Уроки:

- Measure twice, optimize once - всегда начинай с анализа
- Indexes are free (почти) - создавай их смело
- Temp tables are friends - для complex queries
- Execution plan is king - учись его читать
- Test with production-like data - иначе оптимизации могут не сработать

## 🔮 Будущие улучшения

- Query Store для автоматического мониторинга
- Automatic tuning в SQL Server 2017+
- Columnstore indexes для аналитических запросов
- Partitioning больших таблиц
- In-memory OLTP для горячих данных

---

_Эта оптимизация не только ускорила отчет в 30 раз, но и снизила нагрузку на сервер, улучшила user experience и позволила бизнесу принимать решения быстрее. SQL оптимизация — это не магия, а системный подход к анализу и улучшению._
