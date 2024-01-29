import unittest
from OilPricePredictor.Transformer import FloatEmbedding
from OilPricePredictor.Transformer import  AttentionHead
import torch
import random


class FloatEmbeddingTester(unittest.TestCase):
    '''
    Tests to ensure embedding works properly
    '''

    def test_embedding(self):
        '''
        Tests that embedding has propper shape
        :return: None
        '''
        x = torch.Tensor([[3.2, 4.6, 7.8], [2.1, 1.4, 5.2]])
        n_features = 10

        float_embedding = FloatEmbedding.FloatEmbedding(n_features)
        embed_x = float_embedding(x)
        self.assertEqual(embed_x.shape, (2,3,n_features))



