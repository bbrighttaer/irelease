# Author: bbrighttaer
# Project: GPMT
# Date: 5/21/2020
# Time: 9:56 AM
# File: expert_rnn.py

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import joblib
import random
import time
from datetime import datetime as dt

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from soek import Trainer, DataNode, CategoricalParam, RealParam, RandomSearch, \
    BayesianOptSearch
from soek.bopt import GPMinArgs
from torch.utils.data import Dataset

from gpmt.dataloader import load_smiles_data
from gpmt.utils import Flags, time_since, get_fp

currentDT = dt.now()
date_label = currentDT.strftime("%Y_%m_%d__%H_%M_%S")

seeds = [42]


class SmilesDataset(Dataset):
    def __init__(self, x, y):
        self.X, processed_indices, invalid_indices = get_fp(x)
        self.y = y[processed_indices]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item], self.y[item]


class ExpertTrainer(Trainer):
    @staticmethod
    def initialize(hparams, train_data, val_data, test_data):
        train_data = SmilesDataset(*train_data)
        val_data = SmilesDataset(*val_data)
        test_data = SmilesDataset(*test_data)

        # Create model
        model = SVR(kernel=hparams['kernel'], C=hparams['C'], epsilon=hparams['epsilon'])
        model = make_pipeline(StandardScaler(), model)
        metrics = [mean_squared_error, r2_score]

        return {'data': {'train': train_data, 'val': val_data, 'test': test_data}, 'model': model, 'metrics': metrics}

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
    def train(init_dict, n_iterations, transformer, sim_data_node=None, is_hsearch=False, tb_writer=None,
              print_every=10):
        start = time.time()
        model = init_dict['model']
        data = init_dict['data']
        metrics = init_dict['metrics']

        for phase in ['train', 'val' if is_hsearch else 'test']:
            X = data[phase].X
            y_true = data[phase].y.reshape(-1, )
            if phase == 'train':
                model.fit(X, y_true)
            y_pred = model.predict(X).reshape(-1, )
            eval_dict = {}
            score = ExpertTrainer.evaluate(eval_dict,
                                           transformer.inverse_transform(y_true),
                                           transformer.inverse_transform(y_pred),
                                           metrics)
            print(f'{phase}: {eval_dict}')

        print(f'Time elapsed: {time_since(start)}')
        return {'model': model, 'score': score, 'epoch': 1}

    @staticmethod
    def evaluate_model(*args, **kwargs):
        super().evaluate_model(*args, **kwargs)

    @staticmethod
    def save_model(model, path, name):
        os.makedirs(os.path.join(path, 'expert_svr'), exist_ok=True)
        file = os.path.join(path, 'expert_svr', name + ".joblib")
        with open(file, 'wb') as f:
            joblib.dump(model, f)

    @staticmethod
    def load_model(path, name):
        with open(os.path.join(path, name), 'rb') as f:
            return joblib.load(f)


def main(flags):
    mode = 'eval' if flags.eval else 'train'
    sim_label = f'expert_svr_model_{mode}'

    print('--------------------------------------------------------------------------------')
    print(f'{sim_label}\tData file: {flags.data_file}')
    print('--------------------------------------------------------------------------------')

    hparam_search = None

    sim_data = DataNode(label=sim_label, metadata=date_label)
    nodes_list = []
    sim_data.data = nodes_list

    # Load the data
    data_dict, transformer = load_smiles_data(flags.data_file, flags.cv, normalize_y=True, k=flags.folds,
                                              index_col=None)

    for seed in seeds:
        data_node = DataNode(label="seed_%d" % seed)
        nodes_list.append(data_node)

        # ensure reproducibility
        random.seed(seed)
        np.random.seed(seed)

        trainer = ExpertTrainer()
        folds = flags.folds if flags.cv else 1
        if flags.hparam_search:
            print(f'Hyperparameter search enabled: {flags.hparam_search_alg}')
            # arguments to callables
            extra_init_args = {}
            extra_data_args = {'cv': flags.cv,
                               'data': data_dict}
            extra_train_args = {'n_iterations': 5000,
                                'transformer': transformer,
                                'is_hsearch': True,
                                'tb_writer': None}
            hparams_conf = hparams_config()
            if hparam_search is None:
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
                                           results_file=f'{flags.hparam_search_alg}_{sim_label}_{date_label}')
            start = time.time()
            stats = hparam_search.fit()
            print(f'Duration = {time_since(start)}')
            print(stats)
            print("Best params = {}, duration={}".format(stats.best(), time_since(start)))
        else:
            hyper_params = default_params(flags)
            # Initialize the model and other related entities for training.
            if flags.cv:
                folds_data = []
                data_node.data = folds_data
                data_node.label = data_node.label + 'cv'
                for k in range(folds):
                    k_node = DataNode(label="fold-%d" % k)
                    folds_data.append(k_node)
                    start_fold(k_node, data_dict, transformer, flags, hyper_params, trainer, k, None)
            else:
                start_fold(data_node, data_dict, transformer, flags, hyper_params, trainer, folds, None)

    # save simulation data resource tree to file.
    sim_data.to_json(path="./analysis/")


def start_fold(sim_data_node, data_dict, transformer, flags, hyper_params, trainer, k=None, sw_creator=None):
    data = trainer.data_provider(k, data_dict, flags.cv)
    init_args = trainer.initialize(hparams=hyper_params, train_data=data["train"], val_data=data["val"],
                                   test_data=data["test"])
    if flags.eval:
        pass
    else:
        # Train the model
        results = trainer.train(init_args, n_iterations=10000, transformer=transformer, sim_data_node=sim_data_node,
                                tb_writer=sw_creator, print_every=1)
        model, score, epoch = results['model'], results['score'], results['epoch']
        # Save the model.
        label = f'svr_predictor_epoch_{epoch}_{round(score, 3)}'
        if flags.cv:
            label += f'_k{k}'
        trainer.save_model(model, flags.model_dir, label)


def default_params(flag):
    return {'kernel': 'rbf',
            'C': 1.0,
            'epsilon': 0.1}


def hparams_config():
    return {'kernel': CategoricalParam(choices=['linear', 'poly', 'rbf', 'sigmoid']),
            'C': RealParam(min=0.1, max=5),
            'epsilon': RealParam()}


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
