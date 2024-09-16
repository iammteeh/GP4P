from domain.env import USE_DUMMY_DATA, DATA_SLICE_AMOUNT
from adapters.preprocessing import prepare_dataset, preprocessing
import torch

def init_pipeline(use_dummy_data=USE_DUMMY_DATA, extra_features=None, scaler=None, training_size=DATA_SLICE_AMOUNT):
    ds = prepare_dataset(use_dummy_data)
    feature_names, X_train, X_test, y_train, y_test = preprocessing(ds, extra_features, scaler, training_size)

    return ds, feature_names, X_train, X_test, y_train, y_test

def yield_experiments(ds, extra_features=None, scaler=None, training_size=DATA_SLICE_AMOUNT):
    print(f"training_size: {training_size}")
    for batch in training_size:
        feature_names, X_train, X_test, y_train, y_test = preprocessing(ds, extra_ft=None, scaler=None, training_size=batch)

        # slice X_test such that it has the same shape as X_train
        # TODO: this shouldn't be necessary
        if len(X_test) > len(X_train):
            X_test = X_test[:len(X_train)]
            y_test = y_test[:len(X_train)]

        # transform test data to tensor
        X_test = torch.tensor(X_test).float()
        y_test = torch.tensor(y_test).float()

        yield (feature_names, X_train, X_test, y_train, y_test) # return a generator, so that the data is not loaded into memory all at once, but one batch at a time; use next() to get the next batch

def get_numpy_features(X_train, X_test, y_train, y_test):
    # use ndarrays of X and y
    X_train = X_train[1]
    X_test = X_test[1]
    y_train = y_train[1]
    y_test = y_test[1]
    return X_train, X_test, y_train, y_test

def get_tensor_features(X_train, X_test, y_train, y_test):
    X_train = torch.tensor(X_train).double()
    X_test = torch.tensor(X_test).double()
    y_train = torch.tensor(y_train).double().unsqueeze(-1) # make sure y is a column vector
    y_test = torch.tensor(y_test).double().unsqueeze(-1) # make sure y is a column vector
    return X_train, X_test, y_train, y_test