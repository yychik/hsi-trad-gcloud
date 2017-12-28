from preproc import *
from model import *
import datetime
import pandas as pd
import numpy as np
import time
import tensorflow as tf

def generate_data(path, fwd_step, train_sd, eval_sd, test_sd, test_ed, lower_cutoff, upper_cutoff):

    # Function to generate necessary train and valuation data from raw data
    filename = ['d_hsi.csv', 'd_global_indices.csv']

    # Read data
    X = []
    for f in filename:
        print('Reading data from ', (path + f))
        X.append(pd.read_csv(path + f, index_col=0, parse_dates=True))

    X = pd.concat(X, axis=1)

    # Get forward returns
    labels = get_labels_from_fwd_ret(get_fwd_ret(X['CLOSE'], fwd_step), upper_cutoff, lower_cutoff)

    # Split into training and valuation set
    X_scaler, X_train, X_eval, X_test = train_val_test_split(X, train_sd, eval_sd, test_sd, test_ed)
    _, y_train, y_eval, y_test = train_val_test_split(labels, train_sd, eval_sd, test_sd, test_ed, normalize=None)

    return X_scaler, X_train, X_eval, X_test, y_train, y_eval, X_test


if __name__ == '__main__':

    # Parameters for generate data
    path = 'C:/Users/bchik/Udacity/hsi-trad/'
    train_sd = datetime.date(2004, 1, 2)
    eval_sd = datetime.date(2016, 1, 3)
    test_sd = datetime.date(2017, 6, 1)
    test_ed = datetime.date(2017, 12, 21)
    fwd_step = 1
    lower_cutoff = -0.0005
    upper_cutoff = 0.0005

    X_scaler, X_train, X_eval, X_test, y_train, y_eval, y_test = generate_data(path, fwd_step, train_sd, eval_sd, test_sd, test_ed, lower_cutoff, upper_cutoff)
    print(y_train.shape)
    print(X_train.shape)