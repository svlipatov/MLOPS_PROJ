import psycopg2
import pandas as pd
import os
import socket
def get_sales_data_f(top =0):
    # Импорт параметров БД
    if socket.gethostname()[-5:] == 'local':
        os.chdir('/Users/sergei/PycharmProjects/ML_2023_1/Mlops_proj/MLOPS_PROJ/src/')
        df_db = pd.read_csv('db.csv', delimiter=';')
        host = df_db.loc[0, 'host']
    # Docker
    else:
        df_db = pd.read_csv('db.csv', delimiter=';')
        host = df_db.loc[0, 'host2']
    # Соединение
    with psycopg2.connect(dbname=df_db.loc[0,'dbname'], user=df_db.loc[0,'user'],
                            password=df_db.loc[0,'password'], host=host) as conn:
        sql = "SELECT date, sum(sales) as sales FROM public.\"SALES\" group by date"
        if top != 0:
            sql = sql + ' limit ' + str(top)
        # Результат запроса в DataFrame
        df_sales = pd.read_sql(sql, conn)
        return df_sales


if __name__ == "__main__":
    df_sales_data = get_sales_data_f()
    print(df_sales_data)





