# From: https://github.com/isayev/ReLeaSE/blob/master/release/predictor.py
# Included in this project to serve as a proof of concept expert model.

from __future__ import print_function
from __future__ import division

import numpy as np

from sklearn.externals import joblib
from sklearn import metrics

from gpmt.data import PredictorData
from gpmt.utils import get_fp, get_desc, normalize_desc, cross_validation_split, canonical_smiles

from sklearn.ensemble import RandomForestRegressor as RFR


class VanillaQSAR(object):
    def __init__(self, model_instance=None, model_params=None, data_file=None,
                 model_type='classifier', ensemble_size=5, normalization=False):
        super(VanillaQSAR, self).__init__()
        self.model_instance = model_instance
        self.model_params = model_params
        self.ensemble_size = ensemble_size
        self.model = []
        self.normalization = normalization
        if model_type not in ['classifier', 'regressor']:
            raise Exception("model type must be either classifier or regressor")
        self.model_type = model_type
        if isinstance(self.model_instance, list):
            assert (len(self.model_instance) == self.ensemble_size)
            assert (isinstance(self.model_params, list))
            assert (len(self.model_params) == self.ensemble_size)
            for i in range(self.ensemble_size):
                self.model.append(self.model_instance[i](**model_params[i]))
        else:
            for _ in range(self.ensemble_size):
                self.model.append(self.model_instance(**model_params))
        if self.normalization:
            self.desc_mean = [0] * self.ensemble_size
        self.metrics_type = None
        if data_file:
            pred_data = PredictorData(path=data_file, get_features=get_fp)
            print('Fitting expert model...')
            pred_results = self.fit_model(pred_data, cv_split='random')
            print(f'Expert model fit results = {pred_results}')

    def fit_model(self, data, cv_split='stratified'):
        eval_metrics = []
        x = data.x
        if self.model_type == 'classifier' and data.binary_y is not None:
            y = data.binary_y
        else:
            y = data.y
        cross_val_data, cross_val_labels = cross_validation_split(x=x, y=y,
                                                                  split=cv_split,
                                                                  n_folds=self.ensemble_size)
        for i in range(self.ensemble_size):
            train_x = np.concatenate(cross_val_data[:i] +
                                     cross_val_data[(i + 1):])
            test_x = cross_val_data[i]
            train_y = np.concatenate(cross_val_labels[:i] +
                                     cross_val_labels[(i + 1):])
            test_y = cross_val_labels[i]
            if self.normalization:
                train_x, desc_mean = normalize_desc(train_x)
                self.desc_mean[i] = desc_mean
                test_x, _ = normalize_desc(test_x, desc_mean)
            self.model[i].fit(train_x, train_y.ravel())
            predicted = self.model[i].predict(test_x)
            if self.model_type == 'classifier':
                eval_metrics.append(metrics.f1_score(test_y, predicted))
                self.metrics_type = 'F1 score'
            elif self.model_type == 'regressor':
                r2 = metrics.r2_score(test_y, predicted)
                eval_metrics.append(r2)
                self.metrics_type = 'R^2 score'
            else:
                raise RuntimeError()
        return eval_metrics, self.metrics_type

    def load_model(self, path):
        # TODO: add iterable path object instead of static path 
        self.model = []
        for i in range(self.ensemble_size):
            m = joblib.load(path + str(i) + '.pkl')
            self.model.append(m)
        if self.normalization:
            arr = np.load(path + 'desc_mean.npy')
            self.desc_mean = arr

    def save_model(self, path):
        assert self.ensemble_size == len(self.model)
        for i in range(self.ensemble_size):
            joblib.dump(self.model[i], path + str(i) + '.pkl')
        if self.normalization:
            np.save(path + 'desc_mean.npy', self.desc_mean)

    def _predict(self, objects=None, average=True, get_features=None,
                 **kwargs):
        objects = np.array(objects)
        invalid_objects = []
        processed_objects = []
        if get_features is not None:
            x, processed_indices, invalid_indices = get_features(objects,
                                                                 **kwargs)
            processed_objects = objects[processed_indices]
            invalid_objects = objects[invalid_indices]
        else:
            x = objects
        if len(x) == 0:
            processed_objects = []
            prediction = []
            invalid_objects = objects
        else:
            prediction = []
            for i in range(self.ensemble_size):
                m = self.model[i]
                if self.normalization:
                    x, _ = normalize_desc(x, self.desc_mean[i])
                prediction.append(m.predict(x))
            prediction = np.array(prediction)
            if average:
                prediction = prediction.mean(axis=0)
        return processed_objects, prediction, invalid_objects

    def predict(self, inp_smiles):
        sanitized = canonical_smiles(inp_smiles, sanitize=False, throw_warning=False)[:-1]
        unique_smiles = list(np.unique(sanitized))
        smiles, prediction, nan_smiles = self._predict(unique_smiles, get_features=get_fp)
        return smiles, prediction


def get_reward_jak2_max(smiles, predictor, invalid_reward=0.0):
    if isinstance(smiles, str):
        smiles = [smiles]
    mol, prop, nan_smiles = predictor._predict(smiles, get_features=get_fp)
    if len(nan_smiles) == 1:
        return invalid_reward
    return np.exp(prop[0] / 3)


# Model used in https://github.com/isayev/ReLeaSE/blob/master/JAK2_min_max_demo.ipynb
rf_qsar_predictor = VanillaQSAR(model_instance=RFR,
                                model_params={'n_estimators': 250, 'n_jobs': 10},
                                model_type='regressor',
                                data_file='../data/jak2_data.csv')
