import math
import numpy as np
import pandas as pd

class DataLoader():

    def __init__(self, filename, split, cols):
        dataframe = pd.read_csv(filename)
        i_split = int(len(dataframe)*split)
        self.data_train = dataframe.get(cols).values[:i_split]
        self.data_test = dataframe.get(cols).values[i_split:]
        self.len_train = len(self.data_train)
        self.len_test = len(self.data_test)
        self.len_train_windows = None

    def get_test_data(self, seq_len, normalise):

        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])

        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows

        x = data_windows[:, :-1]
        y = data_windows[:, -1, [0]]
        return x, y


    def normalise_windows(self, window_data, single_window=False):
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for windows in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                normalised_col = [((float(p)/float(window[0, col_i])) - 1) for p in window[:, col_i]] 
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T
            normalised_data.append(normalised_window)

        return np.array(normalised_data)

