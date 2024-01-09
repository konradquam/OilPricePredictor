import json

# Function to read data from the JSON file
def read_json(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
        return data
