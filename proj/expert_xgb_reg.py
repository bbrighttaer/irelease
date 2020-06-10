# Author: bbrighttaer
# Project: irelease
# Date: 10/06/20
# Time: 1: AM
# File: expert_xgb_reg.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import pickle
import random
import time
from datetime import datetime as dt

import joblib
import numpy as np
import xgboost as xgb
from soek import *
from soek.bopt import GPMinArgs
from sklearn.metrics import r2_score, mean_squared_error

from irelease.dataloader import load_smiles_data
from irelease.utils import time_since, SmilesDataset

currentDT = dt.now()
date_label = currentDT.strftime("%Y_%m_%d__%H_%M_%S")

seeds = [1]


class XGBExpert(Trainer):

    @staticmethod
    def initialize(hparams, train_dataset, val_dataset, test_dataset):
        train_data = SmilesDataset(*train_dataset)
        val_data = SmilesDataset(*val_dataset) if val_dataset else None
        test_data = SmilesDataset(*test_dataset)

        p = {'objective': hparams['objective'], 'max_depth': hparams['max_depth'],
             'subsample': hparams['subsample'], 'colsample_bytree': hparams['colsample_bytree'],
             'n_estimators': hparams['n_estimators'], 'gamma': hparams['gamma'],
             'reg_lambda': hparams['reg_lambda'], 'learning_rate': hparams['learning_rate'],
             'seed': hparams['seed'], 'eval_metric': 'rmse', 'verbosity': 1}

        # metrics
        metrics = [mean_squared_error, r2_score]
        return p, {'train': train_data, 'val': val_data, 'test': test_data}, metrics

    @staticmethod
    def data_provider(k, data, cv):
        if cv:
            return data[list(data.keys())[k]]
        else:
            return data

    @staticmethod
    def evaluate(eval_out, y_true, y_pred, metrics):
        for metric in metrics:
            res = metric(y_true, y_pred)
            eval_out[metric.__name__] = res
        return eval_out['r2_score']

    @staticmethod
    def train(xgb_params, data, metrics, transformer, n_iters=3000, is_hsearch=False, sim_data_node=None):
        start = time.time()
        metrics_dict = {}

        print('Fitting XGBoost...')
        xgb_eval_results = {}
        dmatrix_train = xgb.DMatrix(data=data['train'].X, label=data['train'].y.reshape(-1, ))
        eval_type = 'val' if is_hsearch else 'test'
        dmatrix_eval = xgb.DMatrix(data=data[eval_type].X, label=data[eval_type].y.reshape(-1, ))
        model = xgb.train(xgb_params, dmatrix_train, n_iters, [(dmatrix_train, 'train'), (dmatrix_eval, eval_type)],
                          early_stopping_rounds=10, evals_result=xgb_eval_results)

        # evaluation
        y_hat = model.predict(dmatrix_eval).reshape(-1, )
        y_true = dmatrix_eval.get_label().reshape(-1, )
        eval_dict = {}
        score = XGBExpert.evaluate(eval_dict, transformer.inverse_transform(y_true),
                                   transformer.inverse_transform(y_hat), metrics)
        for m in eval_dict:
            if m in metrics_dict:
                metrics_dict[m].append(eval_dict[m])
            else:
                metrics_dict[m] = [eval_dict[m]]
        print('Evaluation: score={}, metrics={}'.format(score, eval_dict))

        duration = time.time() - start
        print('\nModel training duration: {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))
        return {'model': model, 'score': score, 'epoch': model.best_iteration}

    @staticmethod
    def evaluate_model(*args, **kwargs):
        super().evaluate_model(*args, **kwargs)

    @staticmethod
    def save_model(model, path, name):
        os.makedirs(os.path.join(path, 'expert_xgb_reg'), exist_ok=True)
        file = os.path.join(path, 'expert_xgb_reg', name + ".joblib")
        with open(file, 'wb') as f:
            joblib.dump(model, f)

    @staticmethod
    def load_model(path, name):
        with open(os.path.join(path, name), 'rb') as f:
            return joblib.load(f)


def main(flags):
    mode = 'eval' if flags.eval else 'train'
    sim_label = f'expert_XGBoost_model_{mode}'

    print('--------------------------------------------------------------------------------')
    print(f'{sim_label}\tData file: {flags.data_file}')
    print('--------------------------------------------------------------------------------')

    sim_data = DataNode(label=sim_label, metadata=date_label)
    nodes_list = []
    sim_data.data = nodes_list

    # Load the data
    data_dict, transformer = load_smiles_data(flags.data_file, flags.cv, normalize_y=True, k=flags.folds, shuffle=5,
                                              create_val=True, train_size=.7, index_col=None)

    for seed in seeds:
        data_node = DataNode(label="seed_%d" % seed)
        nodes_list.append(data_node)

        # ensure reproducibility
        random.seed(seed)
        np.random.seed(seed)

        trainer = XGBExpert()
        folds = flags.folds if flags.cv else 1
        if flags.hparam_search:
            print(f'Hyperparameter search enabled: {flags.hparam_search_alg}')
            # arguments to callables
            extra_init_args = {}
            extra_data_args = {'cv': flags.cv,
                               'data': data_dict}
            extra_train_args = {'n_iters': 5000,
                                'transformer': transformer,
                                'is_hsearch': True}
            hparams_conf = get_hparam_config(flags, seed)
            search_alg = {'random_search': RandomSearch,
                          'bayopt_search': BayesianOptSearch}.get(flags.hparam_search_alg,
                                                                  BayesianOptSearch)
            search_args = GPMinArgs(n_calls=10, random_state=seed)
            hparam_search = search_alg(hparam_config=hparams_conf,
                                       num_folds=folds,
                                       initializer=trainer.initialize,
                                       data_provider=trainer.data_provider,
                                       train_fn=trainer.train,
                                       save_model_fn=trainer.save_model,
                                       alg_args=search_args,
                                       init_args=extra_init_args,
                                       data_args=extra_data_args,
                                       train_args=extra_train_args,
                                       data_node=data_node,
                                       split_label='random',
                                       sim_label=sim_label,
                                       dataset_label=os.path.split(flags.data_file)[1],
                                       results_file=f'{flags.hparam_search_alg}_{sim_label}_'
                                                    f'seed_{seed}_{date_label}')
            start = time.time()
            stats = hparam_search.fit()
            print(f'Duration = {time_since(start)}')
            print(stats)
            print("Best params = {}, duration={}".format(stats.best(), time_since(start)))
        else:
            hyper_params = default_hparams(flags, seed)
            # Initialize the model and other related entities for training.
            if flags.cv:
                folds_data = []
                data_node.data = folds_data
                data_node.label = data_node.label + 'cv'
                for k in range(folds):
                    k_node = DataNode(label="fold-%d" % k)
                    folds_data.append(k_node)
                    start_fold(k_node, data_dict, transformer, flags, hyper_params, trainer, seed, k)
            else:
                start_fold(data_node, data_dict, transformer, flags, hyper_params, trainer, seed, folds)

    # save simulation data resource tree to file.
    sim_data.to_json(path="./analysis/")


def start_fold(sim_data_node, data_dict, transformer, flags, hyper_params, trainer, seed, k):
    data = trainer.data_provider(k, data_dict, flags.cv)
    xgb_params, data, metrics = trainer.initialize(hparams=hyper_params, train_dataset=data["train"],
                                                   val_dataset=data["val"] if 'val' in data else None,
                                                   test_dataset=data["test"])
    if flags.eval:
        pass
    else:
        # Train the model
        results = trainer.train(xgb_params, data, metrics, n_iters=10000, transformer=transformer,
                                sim_data_node=sim_data_node)
        model, score, epoch = results['model'], results['score'], results['epoch']
        # Save the model.
        label = f'xgb_predictor_epoch_seed_{seed}_{epoch}_{round(score, 3)}'
        if flags.cv:
            label += f'_k{k}'
        trainer.save_model(model, flags.model_dir, label)


def default_hparams_rand(flags, seed):
    return {
        "reg_lambda": 0.1,
    }


def default_hparams(flags, seed):
    return {
        'seed': seed,
        'objective': 'reg:squarederror',
        'max_depth': 10,
        'subsample': 0.9662786796693295,
        'colsample_bytree': 0.5640622239646784,
        'n_estimators': 200,
        'gamma': 0.3124800792567785,
        'reg_lambda': 0.45692265456642367,
        'learning_rate': 0.0035615821384587594,
    }


def get_hparam_config(flags, seed):
    return {
        'seed': ConstantParam(seed),
        'objective': ConstantParam('reg:squarederror'),  # reg:linear
        'max_depth': DiscreteParam(5, 10),
        'subsample': RealParam(min=0.5),
        'colsample_bytree': RealParam(min=0.5),
        'n_estimators': DiscreteParam(min=50, max=200),
        'gamma': RealParam(min=0.1),
        'reg_lambda': RealParam(min=0.1),
        'learning_rate': LogRealParam(),
    }


class Flags(object):
    # enables using either object referencing or dict indexing to retrieve user passed arguments of flag objects.
    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        setattr(self, key, value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train an expert model')
    parser.add_argument('--data_file', type=str, help='Dataset file')
    parser.add_argument('--cv', action='store_true', help='Applys cross validation training')
    parser.add_argument('-k', '--folds', dest='folds', default=5, type=int, help='Number of folds for CV if enabled')
    parser.add_argument('--model_dir',
                        type=str,
                        default='./model_dir',
                        help='Directory containing models')
    parser.add_argument("--hparam_search", action="store_true",
                        help='If true, hyperparameter searching would be performed.')
    parser.add_argument('--hparam_search_alg',
                        type=str,
                        default='bayopt_search',
                        help='Hyperparameter search algorithm to use. One of [bayopt_search, random_search]')
    parser.add_argument('--eval',
                        action='store_true',
                        help='If true, a saved model is loaded and evaluated')
    parser.add_argument('--eval_model_name',
                        default=None,
                        type=str,
                        help='The filename of the model to be loaded from the directory specified in --model_dir')

    # Package all arguments
    args = parser.parse_args()
    flags = Flags()
    args_dict = args.__dict__
    for arg in args_dict:
        setattr(flags, arg, args_dict[arg])
    main(flags)
