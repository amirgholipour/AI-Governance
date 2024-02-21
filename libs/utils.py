import json
def read_from_json(file_path):
    """
    Read data from a JSON file at the specified file path and return it.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
def save_as_json(file_path, data):
    """
    Save the given data as a JSON file at the specified file path.
    """
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)