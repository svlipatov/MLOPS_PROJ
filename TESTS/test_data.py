import allure
from SRC.model_fit import calculate_metric
from SRC.model_fit import get_sales_data_f
from SRC.model_fit import get_add_features
from SRC.model_fit import get_preprocessors
from SRC.model_fit import get_model
from SRC.model_fit import split_x_y
from SRC.model_fit import split_train_val
from sklearn.pipeline import Pipeline


@allure.feature("Metrics")
class TestMetrics:
    @allure.title("Проверка метрик")
    @allure.description("""
    Шаг:
    Проверить R2 модели на тренировочных данных
    """)
    @allure.severity(allure.severity_level.NORMAL)
    def test_metrics_r2(self):
        with allure.step("Проверить R2 модели на тренировочных данных"):
            min_r2 = 0.7
            lags = 30
            df_sales = get_sales_data_f()
            # Генерация признаков
            df_dataset = get_add_features(df_sales, lags)
            # Запустим обработку пайплайна
            # Настройка подготовки данных
            preprocessors = get_preprocessors(lags)
            # Модель
            model = get_model()
            # Соберем пайплайн
            pipe = Pipeline([
                ('preprocessors', preprocessors),
                ('model', model)])
            # Разбивка на признаки и целевую переменную
            x, y, dates = split_x_y(df_dataset)
            X_train_l1, X_val_l1, y_train_l1, y_val_l1 = split_train_val(x, y)
            pipe.fit( X_train_l1, y_train_l1)
            r2 = calculate_metric(pipe, X_val_l1, y_val_l1)
            assert r2 > min_r2