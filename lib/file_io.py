import csv
import pickle
## from https://github.com/XinhaoMei/DCASE2021_task6_v2/blob/main/tools/file_io.py

def write_csv_file(csv_obj, file_name):
    """
        Write a list of dict to a csv file.
        :param csv_obj: a list of dict
        :param file_name: the file name
    :return: None
    """
    with open(file_name, 'w') as f:
        writer = csv.DictWriter(f, csv_obj[0].keys())
        writer.writeheader()
        writer.writerows(csv_obj)
    print(f'Write to {file_name} successfully.')


def load_csv_file(file_name):
    """
        Load a csv file to a list of dict.
        :param file_name: the file name
    :return: a list of dict
    """
    with open(file_name, 'r') as f:
        csv_reader = csv.DictReader(f)
        csv_obj = [csv_line for csv_line in csv_reader]
    return csv_obj


def load_pickle_file(file_name):

    with open(file_name, 'rb') as f:
        pickle_obj = pickle.load(f)
    return pickle_obj


def write_pickle_file(obj, file_name):

    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)
    print(f'Write to {file_name} successfully.')
