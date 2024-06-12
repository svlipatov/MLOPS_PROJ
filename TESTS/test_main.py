import allure
from sklearn.utils.validation import check_is_fitted
from src.get_sales_data import get_sales_data_f
from src.model_fit import pipeline_fit
from sklearn.exceptions import NotFittedError

@allure.feature("Prediction")
class TestPrediction:
    @allure.title("Проверка загрузки данных из postgressql")
    @allure.description("""
    Шаг:
    Запросить ограниченное количество данных по продажам из из postgressql
    """)
    @allure.severity(allure.severity_level.NORMAL)
    def get_data(self):
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
    def get_data(self):
        with allure.step("Обучить модель на ограниченном количестве записей"):
            pipeline = pipeline_fit(1000)
            fitted = True
            try:
                check_is_fitted(pipeline)
            except NotFittedError as exc:
                fitted = False
            assert fitted is True