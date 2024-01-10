import os
import pandas as pd
from OilPricePredictor.classes import API_Key_Getter as Key_Getter
from OilPricePredictor.classes import JSON_Reader
import psycopg2

# filepath variables
api_key_path = '../nasdaq_api_key.txt'
db_config_path = '../db_config.json'

# cursor for database connection
cursor = None


def from_api(start_date, end_date, api_key_path='../nasdaq_api_key.txt'):
    '''
    get data from nasdaq data link (tables api)
    :param start_date:
    :param end_date:
    :param api_key_path: filepath to api_key file
    :return: data (pandas dataframe)
    '''
    # read api_key for nasdaq data link
    api_key = Key_Getter.get_api_key(api_key_path)
    # get data from nasdaq export to csv file
    os.system(f'curl "https://data.nasdaq.com/api/v3/datatables/QDL/OPEC.csv?date.gte={start_date}&date.lte={end_date}&api_key={api_key}" > test.csv')
    data = pd.read_csv('test.csv') # put data in data frame
    data = data.rename(columns={'value': 'price'})
    return data


def conn_db(db_config_path=db_config_path):
    '''
    Connects to database
    :param db_config_path: filepath to db config file
    :return: None
    '''
    global cursor

    db_config = JSON_Reader.read_json(db_config_path)
    db_conn = psycopg2.connect(database=db_config['database'], host=db_config['host'], user=db_config['username'], password=db_config['password'], port=db_config['port'])
    cursor = db_conn.cursor()


def insert_db(data):
    '''
    Inserts data into database
    :param data: data to insert (pandas dataframe)
    :return: None
    '''
    for i in range(data.shape[0]):
        cursor.execute('INSERT INTO opec_prices (price_date, price) VALUES (%s, %s)', (data.loc[i, "date"], data.loc[i, "price"]))


def get_db(start_date, end_date):
    '''
    Retrieves data from database
    :param start_date:
    :param end_date:
    :return: data (pandas dataframe)
    '''
    cursor.execute(f'SELECT * FROM opec_prices WHERE price_date >= %s AND price_date <= %s', (start_date, end_date))
    data = pd.DataFrame(cursor.fetchall(), columns=['date', 'price'])
    data['date'] = data['date'].astype(str)
    data['price'] = data['price'].astype(float)
    return data


def delete_db(start_date, end_date):
    '''
    Deletes rows from database opec_prices table
    :param start_date:
    :param end_date:
    :return: None
    '''
    cursor.execute(f'DELETE FROM opec_prices WHERE price_date >= %s AND price_date <= %s', (start_date, end_date))
