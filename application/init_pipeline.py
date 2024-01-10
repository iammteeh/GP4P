from domain.env import USE_DUMMY_DATA
from adapters.preprocessing import prepare_dataset, preprocessing

def init_pipeline(use_dummy_data=USE_DUMMY_DATA, extra_features=None, scaler=None):
    ds = prepare_dataset(use_dummy_data)
    #feature_names = ds.get_feature_names() if not DUMMY_DATA else ds["feature_names"]
    feature_names, X_train, X_test, y_train, y_test = preprocessing(ds, extra_features, scaler)

    return ds, feature_names, X_train, X_test, y_train, y_test

def get_numpy_features(X_train, X_test, y_train, y_test):
    # use ndarrays of X and y
    X_train = X_train[1]
    X_test = X_test[1]
    y_train = y_train[1]
    y_test = y_test[1]
    return X_train, X_test, y_train, y_test