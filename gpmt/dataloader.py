# Author: bbrighttaer
# Project: GPMT
# Date: 5/21/2020
# Time: 9:39 AM
# File: dataloader.py

from __future__ import absolute_import, division, print_function, unicode_literals

import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler


def load_smiles_data(file, cv, normalize_y=True, k=5, header=0, index_col=0, delimiter=',', x_y_cols=(0, 1),
                     reload=True, seed=None, verbose=True):
    assert (os.path.exists(file)), f'File {file} cannot be found.'

    def log(t):
        if verbose:
            print(t)

    data_dict = {}
    transformer = None
    data_dir, filename = os.path.split(file)
    suffix = '_cv' if cv else '_std'
    save_dir = os.path.join(data_dir, filename.split('.')[0] + f'_data_dict{suffix}.joblib')
    trans_save_dir = os.path.join(data_dir, filename.split('.')[0] + f'_transformer{suffix}.joblib')

    # Load data if possible
    if reload and os.path.exists(save_dir):
        log('Loading data...')
        with open(save_dir, 'rb') as f:
            data_dict = joblib.load(f)
        if os.path.exists(trans_save_dir):
            with open(trans_save_dir, 'rb') as f:
                transformer = joblib.load(f)
            log('Data loaded successfully')
            return data_dict, transformer

    # Read and process data
    dataframe = pd.read_csv(file, header=header, index_col=index_col, delimiter=delimiter)
    log(f'Loaded data size = {dataframe.shape}')
    X = dataframe[dataframe.columns[x_y_cols[0]]].values
    y = dataframe[dataframe.columns[x_y_cols[1]]].values.reshape(-1, 1)
    if normalize_y:
        log('Normalizing labels...')
        transformer = RegressionTransformer()
        y = transformer.transform(y)
    log(f'Data directory in use is {data_dir}')

    # Split data
    if cv:
        log(f'Splitting data into {k} folds. Each fold has train, val, and test sets.')
        cv_split = KFold(k, shuffle=True, random_state=seed)
        for i, (train_idx, test_idx) in enumerate(cv_split.split(X, y)):
            x_train, y_train = X[train_idx], y[train_idx]
            x_val, x_test, y_val, y_test = train_test_split(X[test_idx], y[test_idx], test_size=.5, random_state=seed)
            data_dict[f'fold_{i}'] = {'train': (x_train, y_train),
                                      'val': (x_val, y_val),
                                      'test': (x_test, y_test)}
        log('CV splitting completed')
    else:
        log('Splitting data into train, val, and test sets...')
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        x_val, y_val, x_test, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=seed)
        data_dict['train'] = (x_train, y_train)
        data_dict['val'] = (x_val, y_val)
        data_dict['test'] = (x_test, y_test)
        log('Splitting completed.')

    # Persist data if allowed
    if reload:
        with open(save_dir, 'wb') as f:
            joblib.dump(dict(data_dict), f)
        with open(trans_save_dir, 'wb') as f:
            joblib.dump(transformer, f)
    return data_dict, transformer


class RegressionTransformer(object):
    def __init__(self):
        self._scaler = StandardScaler()

    def transform(self, x):
        return self._scaler.fit_transform(x)

    def undo_transform(self, x):
        return self._scaler.inverse_transform(x, copy=True)