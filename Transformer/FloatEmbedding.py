import random

import torch
from torch import nn

class FloatEmbedding(nn.Module):
    def __init__(self, n_features, rand_range=100):
        super(FloatEmbedding, self).__init__()
        self.n_features = n_features
        self.weights = nn.Linear(n_features, n_features)
        self.rand_range = rand_range  # upper bound of random numbers 0-rand_range

    def random_embed(self, input):
        '''
        Embeds continous data into higher dimensions by transforming it into a psuedo-random but deterministic list of floats
        :param input: data batch
        :return: The embedded data (tensor of psuedo-random numbers)
        '''
        random_embedding = []
        for item in input:
            randoms = []
            for seed in item:
                random.seed(seed.item())
                randoms.append(torch.Tensor([random.uniform(0, self.rand_range) for i in range(self.n_features)]))
            randoms = torch.stack(randoms)
            random_embedding.append(randoms)

        random_embedding = torch.stack(random_embedding)
        return random_embedding

    def forward(self, input):
        '''
        Forward pass to embed data with random number generation and multiply by tunable weights
        :param input: input is the batch of data
        :return: Embeddings multiplied by a linear layer of tunable weights
        '''
        random_embedding = self.random_embed(input)
        out = self.weights(random_embedding)

        return out


