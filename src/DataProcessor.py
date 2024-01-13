import pandas as pd
from random import randint
import torch

n_tokens = None # TODO


def set_n_tokens(num):
    '''
    Sets n_tokens to new number
    :param num: number to set n_tokens to
    :return: None
    '''
    global n_tokens
    n_tokens = num

def get_n_tokens():
    '''
    Gets n_tokens
    :return: n_tokens
    '''
    return n_tokens

def chunk_data(data):
    '''
    Breaks the data into chunks of size 2 * (n_tokens + 1) ---
    The plus one is so that all sequences will have a target (n_tokens is context window) ---
    The *2 is so that each chunk contains n_tokens + 1 examples,
    2 specifically is not important but in order to have multiple examples we need to have chunks larger than n_tokens + 1,
    2 seemed like a good choice because it still is able to break data up into many chunks so sampling is more likely to include data from whole timeline (less biased sampling)
    :param data: data to break into chunks
    :return: list of the chunks of data that the data has been broken up into
    '''
    data_size = data.shape[0]  # num rows of data
    chunk_size = 2 * (n_tokens + 1) # important to ensure sample data allowing for several examples per each block, specifically n_tokens examples

    if data_size % chunk_size != 0:
        raise ValueError("chunk_size (2 * (n_tokens + 1)) must divide the number of rows of data")

    chunked_data = [data[i * chunk_size: i * chunk_size + chunk_size] for i in range(data_size // chunk_size)]

    return chunked_data


def train_val_test_split(data, val_size, test_size):
    '''
    Split data into train, validation, and test segments,
    In place manipulates data, adds column for which split it is in
    :param data: data to split
    :param val_size: proportion of data to become val data (decimal)
    :param test_size: proportion of data to become test data (decimal)
    :return: train_data, val_data, test_data
    '''
    try:
        chunked_data = chunk_data(data)
    except ValueError as e:
        raise
    n_test_blocks = int(test_size * len(chunked_data))
    n_val_blocks = int(val_size * len(chunked_data))
    test_data = []
    val_data = []
    train_data = []
    marked_chunks = set() # list of chunks marked as test or val data

    if len(chunked_data) % n_test_blocks != 0:
        raise ValueError("test_size * num_chunks must be an integer")
    if len(chunked_data) % n_val_blocks != 0:
        raise ValueError("val_size * num_chunks must be an integer")

    # add split column ot data
    data['split'] = 'train'

    for i in range(n_test_blocks):
        # get random chunk, add to test_data
        index = randint(0, len(chunked_data) - 1)
        while index in marked_chunks: # ensure we don't resample previously sampled data
            index = randint(0, len(chunked_data) - 1)
        test_data.append(chunked_data[index])

        # mark this chunk as taken with its index, so it is not put into train_Data later
        marked_chunks.add(index)

        # mark sampled data as test data in original data
        start_index = chunked_data[index].index.start
        end_index = chunked_data[index].index.stop
        data.loc[start_index: end_index, 'split'] = 'test'

    for i in range(n_val_blocks):
        # get random chunk, add to val_data
        index = randint(0, len(chunked_data) - 1)
        while index in marked_chunks: # ensure we don't sample test data, or resample previously sampled val data
            index = randint(0, len(chunked_data) - 1)
        val_data.append(chunked_data[index])

        # mark this chunk as taken with its index, so it is not put into train_Data later
        marked_chunks.add(index)

        # mark sampled data as val data in original data
        start_index = chunked_data[index].index.start
        end_index = chunked_data[index].index.stop
        data.loc[start_index: end_index, 'split'] = 'test'

    # train data is all the chunks that were not put into test or val data
    train_data = [chunked_data[i] for i in range(len(chunked_data)) if i not in marked_chunks]

    return train_data, val_data, test_data


def encode(data):
    '''
    data to encode, should be a chunk
    :param data: the data to encode into a tensor (should be in a pandas dataframe with 'price' column)
    :return: tensor of price data
    '''
    return torch.tensor(data['price'].values)






