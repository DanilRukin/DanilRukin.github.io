---
title: "Демо нейросети LeNet на C#"
date: 2024-01-01
description: "Интерактивная демонстрация нейросети LeNet для распознавания рукописных цифр, реализованная на C# и .NET 8"
tags: ["нейросети", "C#", ".NET 8", "LeNet", "демо"]
categories: ["Эксперименты"]
type: "demo"
layout: "demo"
draft: false
---

## 🤖 Интерактивная демонстрация LeNet

Ниже представлено React-приложение, которое позволяет рисовать цифры и видеть предсказания нейросети.

<div id="neural-net-demo">
  <!-- React приложение будет встроено здесь -->
</div>

### Технические детали

**Стек технологий:**

- **Фронтенд:** React 18 + TypeScript + Chakra UI
- **Бэкенд:** .NET 8 Minimal API + ONNX Runtime
- **Модель:** LeNet, обученная на MNIST
- **Инфраструктура:** Docker + GitHub Actions

**Особенности реализации:**

1. **Рисование:** Canvas API с предобработкой изображения
2. **API:** REST endpoint на .NET 8
3. **Модель:** ONNX формат для кроссплатформенности
4. **Визуализация:** Chart.js для отображения активаций слоев

### 📚 Связанные статьи

1. [Архитектура LeNet: от теории к реализации на C#](/blog/lenet-architecture/)
2. [Обучение нейросети на MNIST с помощью ML.NET](/blog/lenet-training/)
3. [Деплой нейросети в продакшен: .NET 8 Web API](/blog/lenet-deployment/)

### 🔧 Исходный код

Все исходники доступны на GitHub:

- [Frontend](https://github.com/DanilRukin/neuralnet-demo-frontend)
- [Backend](https://github.com/DanilRukin/neuralnet-demo-backend)
- [Модель и обучение](https://github.com/DanilRukin/lenet-csharp)

<script type="module" crossorigin src="/demo/assets/index.js"></script>
<link rel="stylesheet" href="/demo/assets/index.css">
