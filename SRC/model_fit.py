import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.model_selection import cross_validate
from sklearn.metrics import r2_score
from SRC.get_sales_data import get_sales_data_f

# Файл одновременно для проекта и для домашних заданий. Много функций чтобы их не дублировать
def make_lags(ts, lags):
    return pd.concat(
        {
            f'y_lag_{i}': ts.shift(i)
            for i in range(1, lags + 1)
        },
        axis=1)
def get_add_features(df_data):
    # Создадим временные лаги
    df_lag1 = make_lags(df_data['sales'], lags=20)
    df_lag1 = df_lag1.fillna(0.0)
    df_lag1 = pd.concat([df_data.shift(1)['date'], df_lag1], axis=1)
    # Оставим только строки, где есть предыдущие значения
    df_lag1 = df_lag1.drop(index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    df_lag1.reset_index(drop=True, inplace=True)
    return df_lag1

def split_x_y(df_data1):
    Y_L1 = df_data1['y_lag_1']
    dates = df_data1['date']
    X_L1 = df_data1.drop(columns=['y_lag_1', 'date'])
    return X_L1, Y_L1, dates
def split_train_val(X, Y):
    # Теперь уже треннировачная и валидационная
    X_train_l1, X_val_l1, y_train_l1, y_val_l1 = train_test_split(X, Y,
                                                                  test_size=0.3,
                                                                  random_state=42)
    return X_train_l1, X_val_l1, y_train_l1, y_val_l1

def get_preprocessors():
    # Стандартизируем показатели
    prep_scale = Pipeline([
        ('scaler', StandardScaler())
    ])
    feat_scale = ['y_lag_2', 'y_lag_3', 'y_lag_4', 'y_lag_5', 'y_lag_6', 'y_lag_7',
                  'y_lag_8', 'y_lag_9', 'y_lag_10', 'y_lag_11', 'y_lag_12', 'y_lag_13',
                  'y_lag_14', 'y_lag_15', 'y_lag_16', 'y_lag_17', 'y_lag_18', 'y_lag_19', 'y_lag_20']

    preprocessors_l1 = ColumnTransformer(transformers=[
        ('prep_scale', prep_scale, feat_scale),
    ])
    return preprocessors_l1

def get_model():
    # Определение модели
    return XGBRegressor()

def cross_validation(X, y, model, cv_rule):
    # Кросс - валидация
    scoring = {'R2': 'r2',
                   '-MSE': 'neg_mean_squared_error',
                   '-MAE': 'neg_mean_absolute_error',
                   '-Max': 'max_error'}
    scores = cross_validate(model,X, y,
                      scoring=scoring, cv=cv_rule )
    DF_score = pd.DataFrame(scores)
    return DF_score.mean()[2:]

def calculate_metric(model_pipe, X, y, metric = r2_score, **kwargs):
    # Метрики на выборке
    y_model = model_pipe.predict(X)
    return metric(y, y_model, **kwargs)

def pipeline_fit(top=0):
    # Получение данных из postgress
    df_sales = get_sales_data_f(top=0)
    # Генерация признаков
    df_dataset = get_add_features(df_sales)
    # Запустим обработку пайплайна
    # Сформируем наборы данных для обучения и проверки
    df_train, df_test = train_test_split(df_dataset,
                                         test_size=0.3,
                                         random_state=42)
    # Настройка подготовки данных
    preprocessors = get_preprocessors()
    # Модель
    model = get_model()
    # Соберем пайплайн
    pipe = Pipeline([
        ('preprocessors', preprocessors),
        ('model', model)])
    # Разбивка на признаки и целевую переменную
    x, y, dates = split_x_y(df_train)
    pipe.fit(x, y)
    return pipe

if __name__ == "__main__":
    # Обучение
    pipeline = pipeline_fit()
