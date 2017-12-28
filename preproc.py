import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import *


def get_fwd_ret(input, steps=1):
    # -----------------------------------------------------#
    # Function to generate forward returns of for label generation
    #
    # INPUTS:
    # inputs = vector of float numbers
    # step = forward step-day return (int)
    #
    # OUTPUTS:
    # data = forward returns (pandas DataFrame)
    # -----------------------------------------------------#

    # Forward Price
    input_fwd = input.shift(-steps)

    # Calculate forward returns
    data = pd.DataFrame(np.log(input_fwd / input))
    data.columns = ['FWD_RET']

    return data


def get_labels_from_fwd_ret(fwd_ret, upper_cutoff, lower_cutoff):
    # -----------------------------------------------------#
    # Function to generate classification labels from forward returns for prediction
    #
    # INPUTS:
    # fwd_ret: dataframe of forward returns (pandas dataframe)
    # upper_cutoff: cutoff % for the classification boundary for "up" label
    # lower_cutoff: cutoff % for classification boundary for "down" label
    #
    # OUTPUT:
    # labels: dataframe of class labels (pandas dataframe)
    # -----------------------------------------------------#

    # Classify the returns
    labels_array = [0 if ret < lower_cutoff else 2 if ret > upper_cutoff else 0 for ret in fwd_ret.values]

    labels = pd.DataFrame(labels_array).set_index(fwd_ret.index)
    labels.columns = ['LABELS']

    return labels

def train_val_test_split(inputs, train_start, val_start, test_start, test_end, normalize='RobustScaler'):
    # -----------------------------------------------------#
    # Function to split data into training, validation and test set by date.
    # This will generate overlapping samples of a step sie feed into LSTM model
    #
    # INPUTS:
    # inputs: pandas array of data to be split, indexed by time
    # train_start: validation set start date
    # val_start: test set start date
    # test_start: test set start date
    # normalize: parameter to control what sklearn normalization to use. It takes whatever sklearn is providing, or None
    #
    # OUTPUT:
    # scaler, (inputs_train, inputs_val, inputs_test): sklearn scaler object, tuples of numpy array
    # -----------------------------------------------------#

    # Split the data
    inputs_train = inputs[train_start:val_start]
    inputs_val = inputs[val_start:test_start]
    inputs_test = inputs[test_start:test_end]

    # Define dictionary to store the corresponding mapping between normalize argument and its scaler object
    dict_scaler = {'MinMaxScaler': MinMaxScaler(),
                   'MaxAbsScaler': MaxAbsScaler(),
                   'StandardScaler': StandardScaler(),
                   'RobustScaler': RobustScaler(),
                   'Normalizer': Normalizer()}

    # Normalize the data
    if normalize != None:

        # Define and fit
        scaler = dict_scaler[normalize]
        scaler.fit(inputs_train)

        # Transform
        inputs_train, inputs_val, inputs_test = scaler.transform(inputs_train), scaler.transform(
            inputs_val), scaler.transform(inputs_test)

        # Output
        return scaler, inputs_train, inputs_val, inputs_test

    else:

        # Output
        return None, inputs_train.as_matrix(), inputs_val.as_matrix(), inputs_test.as_matrix()


def get_batch(inputs, labels, batch_size, steps):
    # -----------------------------------------------------#
    # Function to generate batches inputs to train LSTM model.
    # This will generate overlapping samples of a step sie feed into LSTM model
    #
    # INPUTS:
    # inputs: numpy array of data to be batched
    # batch_size: size of each batch
    # steps: step size to feed into model
    #
    # OUTPUT:
    # batch_data: numpy array of (input_batch, labels_batch) tuple
    # -----------------------------------------------------#

    # Calculate number of sequences with each "steps" step size able to generate
    n_seq = inputs.shape[0] - steps + 1

    # Calculate number of batches
    n_samples = n_seq * steps
    n_batches = n_samples // (batch_size * steps) + 1  # 1 more batch to capture the residual data

    # Error Check: Assert n_batches has to be > 0
    assert (n_batches > 0), 'Not enough data to form 1 batch!'

    # labels
    # labels_batch = labels[:n_batches * batch_size, :]

    # Generate batches
    for n in range(0, n_batches):

        # Container to store the outputs
        inputs_batch = []
        labels_batch = []

        for ii in range(n * batch_size, (n + 1) * batch_size):
            inputs_batch.append(inputs[ii: ii + steps, :])
            labels_batch.append(labels[ii: ii + steps])

        # Return batches
        yield np.stack(inputs_batch), np.stack(labels_batch)

