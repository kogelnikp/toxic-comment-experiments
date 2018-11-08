import pandas as pd

def load_data_from_csv(path, X_cols, Y_cols=None, shuffle=False):
    """ Load data set and return X and Y.
    If Y_cols is None, None is returned for Y.

    Keyword arguments:
    path -- the path to the csv file
    X_cols -- colums which should be treated as features
    Y_cols -- columns which should be treated as labels
    """
    data = pd.read_csv(path, index_col=0, error_bad_lines=False)
    data = data.fillna('')
    if shuffle:
        data = data.sample(frac=1)
    X = data[X_cols]
    if Y_cols is not None:
        Y = data[Y_cols]
        return X, Y
    else:
        return X, None

