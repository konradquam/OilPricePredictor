import unittest
import pandas as pd
from random import randint

import torch

from OilPricePredictor.src.DataProcessor import train_val_test_split
from OilPricePredictor.src.DataProcessor import set_n_tokens
from OilPricePredictor.src.DataProcessor import chunk_data
from OilPricePredictor.src.DataProcessor import encode
from OilPricePredictor.src.DataProcessor import get_batch

class DataProcessorTester(unittest.TestCase):
    '''
    Test splitting data into train, val, and test data ---
    Test dividing data into contiguous chunks, keeping data in order ---
    Test encoding data into pytorch tensor
    '''

    def test_chunking(self):
        '''
        Test dividing data into contiguous chunks, keeping data in order
        :return: None
        '''
        n_tokens = 4
        chunk_size = 10 # 2 * (n_tokens + 1)
        data_size = 500

        set_n_tokens(n_tokens)
        data = pd.DataFrame([randint(0, 100) for i in range(data_size)])
        chunked_data = chunk_data(data)

        self.assertEqual(len(chunked_data), 50)  # correct number of chunks
        self.assertEqual(chunked_data[0].shape[0], 10)  # correct chunk_size
        self.assertEqual(chunked_data[30].shape[0], 10)  # correct chunk_size
        self.assertEqual(chunked_data[0].index.start, 0)  # correct index start
        self.assertEqual(chunked_data[0].index.stop, 10)  # correct index stop
        self.assertEqual(chunked_data[15].index.start, 150)  # correct index start
        self.assertEqual(chunked_data[15].index.stop, 160)  # correct index stop
        self.assertEqual(chunked_data[27].index.start, 270)  # correct index start
        self.assertEqual(chunked_data[27].index.stop, 280)  # correct index stop
        self.assertEqual(chunked_data[49].index.start, 490)  # correct index start
        self.assertEqual(chunked_data[49].index.stop, 500)  # correct index stop

        n_tokens = 5
        set_n_tokens(n_tokens)
        with self.assertRaises(ValueError):
            chunk_data(data)

    def test_splitting(self):
        '''
        Test splitting data into train, val, and test data
        :return: None
        '''
        n_tokens = 4
        data_size = 500
        val_size = 0.1
        test_size = 0.1
        train_size = 1 - val_size - test_size

        set_n_tokens(n_tokens)
        data = pd.DataFrame([randint(0, 100) for i in range(data_size)])
        train, val, test = train_val_test_split(data, val_size, test_size)

        # test that splits are of proper size
        self.assertEqual(5, len(test))
        self.assertEqual(5, len(val))
        self.assertEqual(40, len(train))

        # test that data is not used twice
        # test that none of the indexes from the dataframes are equal to ones in other splits
        for chunk1 in train: # not double data in train and val
            for chunk2 in val:
                self.assertTrue(not chunk1.equals(chunk2))
        for chunk1 in train: # no double data in train and test
            for chunk2 in test:
                self.assertTrue(not chunk1.equals(chunk2))
        for chunk1 in val: # no double data in val and test
            for chunk2 in test:
                self.assertTrue(not chunk1.equals(chunk2))

        # ensure error is raised when using proportions that do not evenly divide the chunks
        # 0.15 * 50 is not an integer
        val_size = 0.15
        test_size = 0.1
        with self.assertRaises(ValueError):
            train_val_test_split(data, val_size, test_size)

        # 0.37 * 50 is not an integer
        val_size = 0.1
        test_size = 0.37
        with self.assertRaises(ValueError):
            train_val_test_split(data, val_size, test_size)

        # val_size to small int(0.01 * 50) = 0
        val_size = 0.01
        test_size = 0.1
        with self.assertRaises(ZeroDivisionError):
            train_val_test_split(data, val_size, test_size)

    def test_encoder(self):
        data = pd.DataFrame({'price': [randint(0, 100) for i in range(100)]})
        data = encode(data)

        self.assertIsInstance(data, torch.Tensor)

    def test_get_batch(self):
        '''
        Test getting a batch of data
        :return: None
        '''
        n_tokens = 4
        data_size = 100
        val_size = 0.1
        test_size = 0.1
        batch_size = 4

        set_n_tokens(n_tokens)
        data = pd.DataFrame({'price': [randint(0, 100) for i in range(data_size)]})
        train, val, test = train_val_test_split(data, val_size, test_size)

        x, y = get_batch(train, batch_size)

        print(x)
        print(y)

        self.assertEqual((batch_size, n_tokens), x.shape)
        self.assertEqual((batch_size, n_tokens), y.shape)
        self.assertEqual([torch.equal(x[i][1:], (y[i][:-1])) for i in range(batch_size)], [True for i in range(batch_size)])



