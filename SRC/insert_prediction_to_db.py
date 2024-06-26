import psycopg2
import pandas as pd
import os
import socket

def single_insert(conn, insert_req):
    """ Execute a single INSERT request """
    cursor = conn.cursor()
    try:
        cursor.execute(insert_req)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1
    cursor.close()
def insert(df_insert):
    # Импорт параметров БД
    if socket.gethostname()[-5:] == 'local':
        os.chdir('/Users/sergei/PycharmProjects/ML_2023_1/Mlops_proj/MLOPS_PROJ/src/')
        df_db = pd.read_csv('db.csv', delimiter=';')
        host = df_db.loc[0, 'host']
    # Docker
    else:
        df_db = pd.read_csv('db.csv', delimiter=';')
        host = df_db.loc[0, 'host2']

    with psycopg2.connect(dbname=df_db.loc[0,'dbname'], user=df_db.loc[0,'user'],
                            password=df_db.loc[0,'password'], host=host) as conn:

        df_insert.reset_index(drop=True, inplace=True)
        for i in df_insert.index:
            query = """
            INSERT into public.predictions(date, sales) values('%s',%s);
            """ % (df_insert.iloc[i,0], df_insert.iloc[i, 1])
            single_insert(conn, query)



