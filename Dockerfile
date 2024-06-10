# Установка базового образа с Python
FROM python:3.9-slim

# включаем root пользователя
user root

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Создание директории приложения
WORKDIR /app

# Копирование файлов зависимостей
COPY requirements.txt .

# Установка зависимостей Python
RUN pip install -r requirements.txt

# Копирование исходного кода приложения в контейнер
COPY . .

