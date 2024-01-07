import torch
import nasdaqdatalink
import os
import pandas as pd
from classes import API_Key_Getter as Key_Getter

# read api_key for nasdaq data link
api_key = Key_Getter.get_api_key()

# get data from nasdaq export to csv file
os.system(f'curl "https://data.nasdaq.com/api/v3/datatables/QDL/OPEC.csv?date.gt=2023-12-15&api_key={api_key}" > test.csv')

# put data in data frame
dataFrame = pd.DataFrame()

dataFrame = pd.read_csv('test.csv')
print(dataFrame)
