import os
import pandas as pd
from OilPricePredictor.classes import API_Key_Getter as Key_Getter
from OilPricePredictor.classes import JSON_Reader
import psycopg2

#filepath variables
api_key_path = '../nasdaq_api_key.txt'
db_config_path = '../db_config.json'

def from_api(start_date, end_date, api_key_path='../nasdaq_api_key.txt'):
    '''
    get data from
    @:param start_date
    to
    @:param end_Date
    '''
    # read api_key for nasdaq data link
    api_key = Key_Getter.get_api_key(api_key_path)
    # get data from nasdaq export to csv file
    os.system(f'curl "https://data.nasdaq.com/api/v3/datatables/QDL/OPEC.csv?date.gte={start_date}&date.lte={end_date}&api_key={api_key}" > test.csv')
    data_frame = pd.read_csv('test.csv') # put data in data frame
    return data_frame

data = from_api('2024-01-02', '2024-01-04')
print(data)


db_config = JSON_Reader.read_json(db_config_path)

db_conn = psycopg2.connect(database=db_config['database'], host=db_config['host'], user=db_config['username'], password=db_config['password'], port=db_config['port'])
cursor = db_conn.cursor()
cursor.execute('SELECT COUNT(price_date) FROM opec_prices')
print(cursor.fetchall())
