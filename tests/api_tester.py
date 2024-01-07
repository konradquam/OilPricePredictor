import unittest
from OilPricePredictor.classes import API_Key_Getter as Key_Getter

class API_Tester(unittest.TestCase):

    # test that getting the apu key from the file == the nasdaq api key
    def test_api_key_getter(self):
        true_api_key = 'your api key' # your api key

        #get api key from file
        api_key = Key_Getter.get_api_key()

        self.assertEqual(api_key, true_api_key)

if __name__ == '__main__':
    unittest.main()