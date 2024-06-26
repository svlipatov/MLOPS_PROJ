import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from SRC.GET_SALES_DATA import get_sales_data_f
import pickle
from SRC.insert_prediction_to_db  import insert

# Файл одновременно для проекта и для домашних заданий. Много функций чтобы их не дублировать
def make_lags(ts, lags):
    return pd.concat(
        {
            f'y_lag_{i}': ts.shift(i)
            for i in range(1, lags + 1)
        },
        axis=1)
def get_add_features(df_data, lags):
    # Создадим временные лаги
    df_data.loc[len(df_data.index)] = df_data.iloc[-1,]
    df_lag1 = make_lags(df_data['sales'], lags)
    df_lag1 = df_lag1.fillna(0.0)
    df_lag1 = pd.concat([df_data.shift(1)['date'], df_lag1], axis=1)
    # Оставим только строки, где есть предыдущие значения
    df_lag1 = df_lag1.drop(index=range(lags))
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

def get_preprocessors(lags):
    # Стандартизируем показатели
    prep_scale = Pipeline([
        ('scaler', StandardScaler())
    ])
    feat_scale = []
    for i in range(2, lags + 1):
        feat_scale.append('y_lag_' + str(i))

    preprocessors_l1 = ColumnTransformer(transformers=[
        ('prep_scale', prep_scale, feat_scale),
    ])
    return preprocessors_l1

def get_model():
    # Определение модели
    return XGBRegressor()

def calculate_metric(model_pipe, X, y, metric = r2_score, **kwargs):
    # Метрики на выборке
    y_model = model_pipe.predict(X)
    return metric(y, y_model, **kwargs)

def pipeline_fit(lags):
    # Получение данных из postgress
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
    pipe.fit(x, y)
    return pipe, df_dataset

def model_save(pipe):
    pkl_filename = "model/model.pkl"
    with open(pkl_filename, 'wb') as file1:
        pickle.dump(pipe, file1)

def predict(pipe, df_pr_data, days_to_predict):
    # Построчно прогнозируем, чтобы результат предыдущего прогноза использовать как признак
    df_pr_data['date'] = df_pr_data['date'].astype("datetime64[ns]").apply(lambda x: x.date())
    max_date = df_pr_data['date'].max()
    for day in range(days_to_predict):
        ts_new = df_pr_data.iloc[len(df_pr_data.index) - 1]
        ts_new = ts_new.shift(1)
        ts_new['date'] = max_date + pd.Timedelta(days=day + 1)
        # В dataframe для прогноза
        df_new = pd.DataFrame(columns = df_pr_data.columns)
        df_new.loc[len(df_new.index)] = ts_new
        # Разбивка на признаки и целевую переменную
        x, y, dates = split_x_y(df_new)
        y = pipe.predict(x)
        ts_new['y_lag_1'] = y
        df_pr_data.loc[len(df_pr_data.index)] = ts_new
    df_res = pd.DataFrame()
    df_res['date'] = df_pr_data.iloc[-5:,0]
    df_res['sales'] = df_pr_data.iloc[-5:, 1]
    return df_res


if __name__ == "__main__":
    # Сколько дней необходимо спрогнозировать
    days_to_predict = 5
    # Количество лаг
    lags = 30
    # Обучение
    pipeline, df_train_data = pipeline_fit(lags)
    # Сохранение модели
    model_save(pipeline)
    # Построение прогноза
    df_predict = predict(pipeline, df_train_data, days_to_predict)
    # Вставка данных прогноза в БД
    insert(df_predict)



