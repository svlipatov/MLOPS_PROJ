import psycopg2
import pandas as pd
def get_sales_data_f():
    # Импорт параметров БД
    df_db = pd.read_csv('/Users/sergei/PycharmProjects/ML_2023_1/DB/db.csv')
    # Соединение
    with psycopg2.connect(dbname=df_db.loc[0,'dbname'], user=df_db.loc[0,'user'],
                            password=df_db.loc[0,'password'], host=df_db.loc[0,'host']) as conn:
        sql = "SELECT date, sum(sales) as sales FROM public.\"SALES\" group by date"
        # Результат запроса в DataFrame
        df_sales = pd.read_sql(sql, conn)
        return df_sales

df_sales_data = get_sales_data_f()
if __name__ == "__main__":
    print(df_sales_data)





