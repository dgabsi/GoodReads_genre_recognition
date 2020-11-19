import numpy as np
import pandas as pd
import os
import pickle


def dump_datasets_to_pickle(data_path, data, pickle_file_name):
    with open(os.path.join(data_path, pickle_file_name), 'wb') as f:
        pickle.dump(data, f)
    f.close()


def load_datasets_from_pickle(data_path, pickle_file_name):
    with open(os.path.join(data_path, pickle_file_name), 'rb') as f:
        data = pickle.load(f)
    f.close()

    return data
