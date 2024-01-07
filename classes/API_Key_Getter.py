from abc import abstractmethod
'''
Gets API key for Nasdaq Data Link
'''
def get_api_key(file_path='../nasdaq_api_key.txt'):
    file_path = file_path
    file = open(file_path, 'r')
    api_key = file.readline()
    file.close()
    return api_key

