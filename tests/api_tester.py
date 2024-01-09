import unittest
from OilPricePredictor.classes import API_Key_Getter as Key_Getter
from OilPricePredictor.src import DataGetter


class API_Tester(unittest.TestCase):

    # test that getting the apu key from the file == the nasdaq api key
    def test_api_key_getter(self):
        true_api_key = 'your api key' # your api key

        #get api key from file
        api_key = Key_Getter.get_api_key()

        self.assertEqual(api_key, true_api_key)

    # get the correct data from the nasdaq api
    # correct start and end dates, and count
    def test_correct_data(self):
        start_date = '2024-01-02'
        end_date = '2024-01-04'
        data = DataGetter.from_api(start_date, end_date)

        self.assertEqual(data.at[2, 'date'], start_date)
        self.assertEqual(data.at[0, 'date'], end_date)
        self.assertEqual(data.shape[0], 3)


if __name__ == '__main__':
    unittest.main()