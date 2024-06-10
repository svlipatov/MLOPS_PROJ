# MLOPS_PROJ
# Цель разработки  
Цель данной разработки - разработать функционал по прогнозированию продаж будущих периодов, построение план-факт отчетности по продажам  
  
# Функционал разработки  
- Хранение набора данных по продажам в БД postgresql
- Автоматизированное формирование экспортной витрины данных в БД postgresql.  
- Получение данных из БД, предобработка этих данных в python и построение прогноза на основе этих данных
- Передача прогнозных данных в БД и построение отчетности план-факт, на основе этих данных и фактических данных
- Хранение сводной информации по каждому расчету: версия данных в БД, версия модели, метрики
  
# Список инструментов, задейстованных в разработке  
- Github - Контроль версий кода разработки  
- БД PostgreSql 16 - Хранение набора данных
- Python 3.11 - предобработка данных и получение прогноза  
- Jenkins - Автоматизация сборки
- Apache airflow - Планировщик загрузок
- Docker - инструмент для развертывания контейнер с независимой средой 

# Развертывание среды
- Jenkins копирует проект с github
- Jenkins разворачивает контейнер docker (с созданием виртуальной среды с библиотекми из файла  requirements, с копированием файлов проекта)

# Особенности проекта

## Логирование изменений данных, DVC
Есть требование по использованию DVC, но для реализуемого проекта он не подходит. Причина в том, что каждая версия DVC хранит полный объем информации. В то время как разные версии фактических данных для прогноза будут отличаться только новыми периодами, а исторические данные будут оставаться теми же. Таким образом DVC будет хранить избыточную информацию и для большого набора данных это будет критичным. Вместо DVC будет использоваться таблица в postgresql, где версионирование данных будет выполняться инкрементально. В отдельной таблице будет храниться соответствие версии данных, модели обучения, метрик. Пример выполнения задания с DVC, есть в LAB4

## Обучение модели и прогноз значений
Задача подразумевает, что каждый месяц будет поступать факт за предыдущий месяц и с учетом него строится прогноз на следующий месяц. Это значит, что каждый месяц модель будет заново обучаться и сразу будет использоваться метод построения прогноза.

# Набор данных  
Набор данных представляет собой продажи розничного магазина в разрезе дней и групп товаров.

# Автор проекта
| Участник      |            Почта               |
|:--------------|:------------------------------:|
| Липатов С.В.  |   svlipatov@bk.ru              |
