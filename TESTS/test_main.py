import allure
from sklearn.utils.validation import check_is_fitted
from SRC.GET_SALES_DATA import get_sales_data_f
from SRC.model_fit import pipeline_fit
from SRC.model_fit import predict
from sklearn.exceptions import NotFittedError


@allure.feature("Prediction")
class TestPrediction:
    @allure.title("Проверка загрузки данных из postgressql")
    @allure.description("""
    Шаг:
    Запросить ограниченное количество данных по продажам из из postgressql
    """)
    @allure.severity(allure.severity_level.NORMAL)
    def test_data_receiving(self):
        with allure.step("Запросить ограниченное количество данных по продажам из из postgressql"):
            df_sales_test = get_sales_data_f(1000)
            assert len(df_sales_test) != 0

    @allure.title("Проверка работы обучения модели")
    @allure.description("""
    Обучение модели на ограниченном количестве записей

    Шаг:
    Обучить модель на ограниченном количестве записей
    """)
    @allure.severity(allure.severity_level.NORMAL)
    def test_model_training(self):
        with allure.step("Обучить модель на ограниченном количестве записей"):
            pipeline, df_train_data = pipeline_fit(1000)
            fitted = True
            try:
                check_is_fitted(pipeline)
            except NotFittedError as exc:
                fitted = False
            assert fitted is True

    @allure.title("Проверка работы предсказаний модели")
    @allure.description("""
    Предсказать целевую переменную
    
    Шаг:
    Предсказать целевую переменную
    """)
    @allure.severity(allure.severity_level.NORMAL)
    def test_predict(self):
        with allure.step("Предсказать целевую переменную"):
            # Сколько дней необходимо спрогнозировать
            days_to_predict = 1
            # Количество лаг
            lags = 5
            # Обучение
            pipeline, df_train_data = pipeline_fit(lags)
            # Построение прогноза
            df_predict = predict(pipeline, df_train_data, days_to_predict)
            assert df_predict.iloc[0,1] != 0

