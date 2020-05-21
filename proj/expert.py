# Author: bbrighttaer
# Project: GPMT
# Date: 5/21/2020
# Time: 9:56 AM
# File: expert.py

from __future__ import absolute_import, division, print_function, unicode_literals

import contextlib
import os
import argparse
import time
import copy
from collections import defaultdict
from datetime import datetime as dt
import random
import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error
from soek.bopt import GPMinArgs
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader
from soek import Trainer, DataNode, CategoricalParam, DiscreteParam, RealParam, LogRealParam, RandomSearch, \
    BayesianOptSearch
from tqdm import tqdm

from gpmt.model import RNNPredictorModel
from gpmt.supervised.dataloader import load_smiles_data
from gpmt.utils import Flags, get_default_tokens, parse_optimizer, time_since

currentDT = dt.now()
date_label = currentDT.strftime("%Y_%m_%d__%H_%M_%S")

seeds = [42]

if torch.cuda.is_available():
    dvc_id = 2
    use_cuda = True
    device = f'cuda:{dvc_id}'
    torch.cuda.set_device(dvc_id)
else:
    device = 'cpu'
    use_cuda = None


class SmilesDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item], self.y[item]


class ExpertTrainer(Trainer):
    @staticmethod
    def initialize(hparams, train_data, val_data, test_data):
        # Create pytorch data loaders
        train_loader = DataLoader(SmilesDataset(train_data[0], train_data[1]),
                                  batch_size=hparams['batch'],
                                  collate_fn=lambda x: x)
        val_loader = DataLoader(SmilesDataset(val_data[0], val_data[1]),
                                batch_size=hparams['batch'],
                                collate_fn=lambda x: x)
        test_loader = DataLoader(SmilesDataset(test_data[0], test_data[1]),
                                 batch_size=hparams['batch'],
                                 collate_fn=lambda x: x)
        # Create model and optimizer
        model = RNNPredictorModel(d_model=hparams['d_model'],
                                  tokens=get_default_tokens(),
                                  num_layers=hparams['rnn_num_layers'],
                                  dropout=hparams['dropout'],
                                  bidirectional=hparams['is_bidirectional'],
                                  unit_type=hparams['unit_type'],
                                  device=device).to(device)
        optimizer = parse_optimizer(hparams, model)
        metrics = [mean_squared_error, r2_score]

        return {'data_loaders': {'train': train_loader,
                                 'val': val_loader,
                                 'test': test_loader},
                'model': model,
                'optimizer': optimizer,
                'metrics': metrics}

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
        predictor = init_dict['model']
        data_loaders = init_dict['data_loaders']
        optimizer = init_dict['optimizer']
        metrics = init_dict['metrics']
        lr_sch = ExponentialLR(optimizer, gamma=0.98)
        if tb_writer:
            tb_writer = tb_writer()
        best_model_wts = predictor.state_dict()
        best_score = -10000
        best_epoch = -1
        terminate_training = False
        n_epochs = n_iterations // len(data_loaders['train'])
        criterion = torch.nn.MSELoss()

        # Since during hyperparameter search values that could cause CUDA memory exception could be sampled
        # we want to ignore such values and find others that are workable within the memory constraints.
        with contextlib.suppress(Exception if not is_hsearch else DummyException):
            a = 1 / 0
            for epoch in range(n_epochs):
                eval_scores = []
                for phase in ['train', 'val', 'test']:
                    if phase == 'train':
                        predictor.train()
                    else:
                        predictor.eval()

                    losses = []
                    metrics_dict = defaultdict(list)
                    for batch in tqdm(data_loaders[phase], desc=f'Phase: {phase}, epoch={epoch + 1}/{n_epochs}'):
                        batch = np.array(batch)
                        x = batch[:, 0]
                        y_true = batch[:, 1]
                        with torch.set_grad_enabled(phase == 'train'):
                            y_true = torch.from_numpy(y_true.reshape(-1, 1).astype(np.float)).float().to(device)
                            y_pred = predictor(x)
                            loss = criterion(y_pred, y_true)
                            losses.append(loss.item())

                        # Perform evaluation using the given metrics
                        eval_dict = {}
                        score = ExpertTrainer.evaluate(eval_dict, y_true.cpu().detach().numpy(),
                                                       transformer.undo_transform(y_pred.cpu().detach().numpy()),
                                                       metrics)
                        for m in eval_dict:
                            if m in metrics:
                                metrics_dict[m].append(eval_dict[m])
                            else:
                                metrics_dict[m] = [eval_dict[m]]

                        # Update weights
                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                        else:
                            eval_scores.append(score)
                    metrics_dict = {k: np.mean(metrics_dict[k]) for k in metrics_dict}
                    if epoch % print_every == 0:
                        print(f'{phase}: epoch={epoch + 1}/{n_epochs}, loss={np.mean(losses)}, metrics={metrics_dict}')
                    if phase == 'train':
                        lr_sch.step()
                # Checkpoint
                score = np.mean(eval_scores)
                if score > best_score:
                    best_score = score
                    best_model_wts = copy.deepcopy(predictor.state_dict())
                    best_epoch = epoch
        predictor.load_state_dict(best_model_wts)
        print(f'Time elapsed: {time_since(start)}')
        return {'model': predictor, 'score': best_score, 'epoch': best_epoch}

    @staticmethod
    def evaluate_model(*args, **kwargs):
        super().evaluate_model(*args, **kwargs)

    @staticmethod
    def save_model(model, path, name):
        os.makedirs(os.path.join(path, 'expert'), exist_ok=True)
        file = os.path.join(path, 'expert', name + ".mod")
        torch.save(model.state_dict(), file)

    @staticmethod
    def load_model(path, name):
        return torch.load(os.path.join(path, name), map_location=torch.device(device))


class DummyException(RuntimeError):
    """ Method or function hasn't been implemented yet. """

    def __init__(self, *args, **kwargs):  # real signature unknown
        pass


def main(flags):
    mode = 'eval' if flags.eval else 'train'
    sim_label = f'expert_model_{mode}'

    print('--------------------------------------------------------------------------------')
    print(f'{device}\n{sim_label}\tData file: {flags.data_file}')
    print('--------------------------------------------------------------------------------')

    hparam_search = None

    sim_data = DataNode(label=sim_label, metadata=date_label)
    nodes_list = []
    sim_data.data = nodes_list

    # Load the data
    data_dict, transformer = load_smiles_data(flags.data_file, flags.cv, normalize_y=True, k=flags.folds)

    for seed in seeds:
        data_node = DataNode(label="seed_%d" % seed)
        nodes_list.append(data_node)

        # ensure reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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
            stats = hparam_search.fit()
            print(stats)
            print("Best params = {}".format(stats.best()))
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
                                tb_writer=sw_creator)
        model, score, epoch = results['model'], results['score'], results['epoch']
        # Save the model.
        label = f'rnn_predictor_epoch_{epoch}_{round(score, 3)}'
        if flags.cv:
            label += f'_k{k}'
        trainer.save_model(model, flags.model_dir, label)


def default_params(flag):
    return {'batch': 128,
            'd_model': 128,
            'rnn_num_layers': 2,
            'dropout': 0.8,
            'is_bidirectional': False,
            'unit_type': 'lstm',
            'optimizer': 'adam',
            # 'optimizer__global__weight_decay': 0.0005,
            'optimizer__global__lr': 0.005}


def hparams_config():
    return {'batch': CategoricalParam(choices=[32, 64, 128]),
            'd_model': DiscreteParam(min=32, max=256),
            'rnn_num_layers': DiscreteParam(min=1, max=3),
            'dropout': RealParam(min=0., max=0.8),
            'is_bidirectional': CategoricalParam(choices=[True, False]),
            'unit_type': CategoricalParam(choices=['gru', 'lstm']),
            'optimizer': CategoricalParam(choices=['sgd', 'adam', 'adadelta', 'adagrad', 'adamax', 'rmsprop']),
            'optimizer__global__weight_decay': LogRealParam(),
            'optimizer__global__lr': LogRealParam()}


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
