import unittest
import pandas as pd
from OilPricePredictor.src import DataGetter


class DB_Tester(unittest.TestCase):
    '''
    Test that the database can be connected to,
    Test inserting data,
    Test retrieving data
    '''

    def test_insert_retrieve_delete(self):
        '''
        Test connecting to database,
        Test inserting data,
        Test retrieving data,
        Test deleting data,
        ---
        All in one function because I need to retrieve inserted data to ensure correct data was inserted,
        I also need to delete data so I don't leave fake data in db
        :return: None
        '''
        start_date = '3000-01-01'
        end_date = '3000-01-03'
        ins_data = pd.DataFrame([(start_date, '{:.2f}'.format(0)), ('3000-01-02', '{:.2f}'.format(0)), (end_date, '{:.2f}'.format(0))], columns=['date', 'price']) #inserted data
        ins_data['date'] = ins_data['date'].astype(str)
        ins_data['price'] = ins_data['price'].astype(float)

        DataGetter.conn_db()
        DataGetter.insert_db(ins_data)

        # retrieved all the inserted data, should be equal
        retrieved_data = DataGetter.get_db(start_date, end_date)
        self.assertTrue(ins_data.equals(retrieved_data))

        # retrieved only one row of the inserted data, should not be equal
        retrieved_data = DataGetter.get_db(start_date, start_date)
        self.assertTrue(not ins_data.equals(retrieved_data))

        # retrieving data not in db, should not be equal
        # retrieved data is an empty dataframe
        retrieved_data = DataGetter.get_db('3000-02-01', '3000-02-03')
        self.assertTrue(not ins_data.equals(retrieved_data))
        self.assertTrue(retrieved_data.empty)

        # delete data, should not be equal
        # retrieved data is an empty dataframe
        DataGetter.delete_db(start_date, end_date)
        retrieved_data = DataGetter.get_db(start_date, end_date)
        self.assertTrue(not ins_data.equals(retrieved_data))
        self.assertTrue(retrieved_data.empty)
