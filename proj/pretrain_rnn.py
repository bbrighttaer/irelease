# Author: bbrighttaer
# Project: GPMT
# Date: 3/23/2020
# Time: 12:03 PM
# File: pretrain.py

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import copy
import math
import os
import random
import time
from datetime import datetime as dt

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from soek import CategoricalParam, LogRealParam, RealParam, DiscreteParam, DataNode, RandomSearch, \
    BayesianOptSearch
from soek.bopt import GPMinArgs
from soek.template import Trainer
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from gpmt.data import GeneratorData
from gpmt.model import Encoder, StackRNN, StackRNNLinear, StackedRNNLayerNorm, StackedRNNDropout
from gpmt.utils import Flags, parse_optimizer, ExpAverage, GradStats, Count, init_hidden, init_cell, init_stack, \
    generate_smiles, time_since, get_default_tokens, canonical_smiles

currentDT = dt.now()
date_label = currentDT.strftime("%Y_%m_%d__%H_%M_%S")

seeds = [1]

if torch.cuda.is_available():
    dvc_id = 0
    use_cuda = True
    device = f'cuda:{dvc_id}'
    torch.cuda.set_device(dvc_id)
else:
    device = 'cpu'
    use_cuda = None


class GpmtPretrain(Trainer):
    @staticmethod
    def initialize(hparams, gen_data, *args, **kwargs):
        gen_data.set_batch_size(hparams['batch_size'])
        # Create main model
        encoder = Encoder(vocab_size=gen_data.n_characters, d_model=hparams['d_model'],
                          padding_idx=gen_data.char2idx[gen_data.pad_symbol],
                          dropout=hparams['dropout'], return_tuple=True)
        # Create RNN layers
        rnn_layers = []
        has_stack = True
        for i in range(1, hparams['num_layers'] + 1):
            rnn_layers.append(StackRNN(layer_index=i,
                                       input_size=hparams['d_model'],
                                       hidden_size=hparams['d_model'],
                                       has_stack=has_stack,
                                       unit_type=hparams['unit_type'],
                                       stack_width=hparams['stack_width'],
                                       stack_depth=hparams['stack_depth'],
                                       k_mask_func=encoder.k_padding_mask))
            if i + 1 < hparams['num_layers']:
                rnn_layers.append(StackedRNNDropout(hparams['dropout']))
                rnn_layers.append(StackedRNNLayerNorm(hparams['d_model']))

        model = nn.Sequential(encoder,
                              *rnn_layers,
                              StackRNNLinear(out_dim=gen_data.n_characters,
                                             hidden_size=hparams['d_model'],
                                             bidirectional=False,
                                             # encoder=encoder,
                                             # dropout=hparams['dropout'],
                                             bias=True))
        if use_cuda:
            model = model.cuda()
        optimizer = parse_optimizer(hparams, model)
        rnn_args = {'num_layers': hparams['num_layers'],
                    'hidden_size': hparams['d_model'],
                    'num_dir': 1,
                    'device': device,
                    'has_stack': has_stack,
                    'has_cell': hparams['unit_type'] == 'lstm',
                    'stack_width': hparams['stack_width'],
                    'stack_depth': hparams['stack_depth']}
        return model, optimizer, gen_data, rnn_args

    @staticmethod
    def data_provider(k, flags):
        tokens = get_default_tokens()
        gen_data = GeneratorData(training_data_path=flags.data_file,
                                 delimiter='\t',
                                 cols_to_read=[0],
                                 keep_header=True,
                                 pad_symbol=' ',
                                 max_len=120,
                                 tokens=tokens,
                                 use_cuda=use_cuda)
        return {"train": gen_data, "val": gen_data, "test": gen_data}

    @staticmethod
    def evaluate(eval_dict, predictions, labels):
        y_true = labels.cpu().detach().numpy()
        y_pred = torch.max(predictions, dim=-1)[1]
        y_pred = y_pred.cpu().detach().numpy()
        acc = accuracy_score(y_true, y_pred)
        eval_dict['accuracy'] = acc
        return acc

    @staticmethod
    def train(model, optimizer, gen_data, rnn_args, n_iters=5000, sim_data_node=None, epoch_ckpt=(1, 2.0),
              tb_writer=None, is_hsearch=False):
        tb_writer = None  # tb_writer()
        start = time.time()
        best_model_wts = model.state_dict()
        best_score = -10000
        best_epoch = -1
        terminate_training = False
        e_avg = ExpAverage(.01)
        num_batches = math.ceil(gen_data.file_len / gen_data.batch_size)
        n_epochs = math.ceil(n_iters / num_batches)
        grad_stats = GradStats(model, beta=0.)

        # learning rate decay schedulers
        # scheduler = sch.StepLR(optimizer, step_size=500, gamma=0.01)

        # pred_loss functions
        criterion = nn.CrossEntropyLoss(ignore_index=gen_data.char2idx[gen_data.pad_symbol])

        # sub-nodes of sim data resource
        loss_lst = []
        train_loss_node = DataNode(label="train_loss", data=loss_lst)
        metrics_dict = {}
        metrics_node = DataNode(label="validation_metrics", data=metrics_dict)
        train_scores_lst = []
        train_scores_node = DataNode(label="train_score", data=train_scores_lst)
        scores_lst = []
        scores_node = DataNode(label="validation_score", data=scores_lst)

        # add sim data nodes to parent node
        if sim_data_node:
            sim_data_node.data = [train_loss_node, train_scores_node, metrics_node, scores_node]

        try:
            # Main training loop
            tb_idx = {'train': Count(), 'val': Count(), 'test': Count()}
            epoch_losses = []
            epoch_scores = []
            for epoch in range(n_epochs):
                phase = 'train'

                # Iterate through mini-batches
                # with TBMeanTracker(tb_writer, 10) as tracker:
                with grad_stats:
                    for b in trange(0, num_batches, desc=f'{phase} in progress...'):
                        inputs, labels = gen_data.random_training_set()
                        batch_size, seq_len = inputs.shape[:2]
                        optimizer.zero_grad()

                        # track history if only in train
                        with torch.set_grad_enabled(phase == "train"):
                            # Create hidden states for each layer
                            hidden_states = []
                            for _ in range(rnn_args['num_layers']):
                                hidden = init_hidden(num_layers=1, batch_size=batch_size,
                                                     hidden_size=rnn_args['hidden_size'],
                                                     num_dir=rnn_args['num_dir'], dvc=rnn_args['device'])
                                if rnn_args['has_cell']:
                                    cell = init_cell(num_layers=1, batch_size=batch_size,
                                                     hidden_size=rnn_args['hidden_size'],
                                                     num_dir=rnn_args['num_dir'], dvc=rnn_args['device'])
                                else:
                                    cell = None
                                if rnn_args['has_stack']:
                                    stack = init_stack(batch_size, rnn_args['stack_width'],
                                                       rnn_args['stack_depth'], dvc=rnn_args['device'])
                                else:
                                    stack = None
                                hidden_states.append((hidden, cell, stack))
                            # forward propagation
                            outputs = model([inputs] + hidden_states)
                            predictions = outputs[0]
                            predictions = predictions.permute(1, 0, -1)
                            predictions = predictions.contiguous().view(-1, predictions.shape[-1])
                            labels = labels.contiguous().view(-1)

                            # calculate loss
                            loss = criterion(predictions, labels)

                        # metrics
                        eval_dict = {}
                        score = GpmtPretrain.evaluate(eval_dict, predictions, labels)

                        # TBoard info
                        # tracker.track("%s/loss" % phase, loss.item(), tb_idx[phase].IncAndGet())
                        # tracker.track("%s/score" % phase, score, tb_idx[phase].i)
                        # for k in eval_dict:
                        #     tracker.track('{}/{}'.format(phase, k), eval_dict[k], tb_idx[phase].i)

                        # backward pass
                        loss.backward()
                        optimizer.step()

                        # for epoch stats
                        epoch_losses.append(loss.item())

                        # for sim data resource
                        train_scores_lst.append(score)
                        loss_lst.append(loss.item())

                        # for epoch stats
                        epoch_scores.append(score)

                        print("\t{}: Epoch={}/{}, batch={}/{}, "
                              "pred_loss={:.4f}, accuracy: {:.2f}, sample: {}".format(time_since(start),
                                                                                      epoch + 1, n_epochs,
                                                                                      b + 1,
                                                                                      num_batches,
                                                                                      loss.item(),
                                                                                      eval_dict['accuracy'],
                                                                                      generate_smiles(
                                                                                          generator=model,
                                                                                          gen_data=gen_data,
                                                                                          init_args=rnn_args,
                                                                                          num_samples=1)
                                                                                      ))
                # End of mini=batch iterations.
        except RuntimeError as e:
            print(str(e))

        duration = time.time() - start
        print('\nModel training duration: {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))
        return {'model': model, 'score': round(np.mean(epoch_scores), 3), 'epoch': n_epochs}

    @staticmethod
    @torch.no_grad()
    def evaluate_model(model, gen_data, rnn_args, sim_data_node=None, num_smiles=1000):
        start = time.time()
        model.eval()

        # Samples SMILES
        samples = []
        step = 100
        count = 0
        for _ in range(int(num_smiles / step)):
            samples.extend(generate_smiles(generator=model, gen_data=gen_data, init_args=rnn_args,
                                           num_samples=step, is_train=False, verbose=True))
            count += step
        res = num_smiles - count
        if res > 0:
            samples.extend(generate_smiles(generator=model, gen_data=gen_data, init_args=rnn_args,
                                           num_samples=res, is_train=False, verbose=True))
        smiles, valid_vec = canonical_smiles(samples)
        valid_smiles = []
        for idx, sm in enumerate(smiles):
            if len(sm) > 0:
                valid_smiles.append(sm)
        v = len(valid_smiles)
        valid_smiles = list(set(valid_smiles))
        print(f'Percentage of valid SMILES = {float(len(valid_smiles)) / float(len(samples)):.2f}, '
              f'Num. samples = {len(samples)}, Num. valid = {len(valid_smiles)}, '
              f'Num. requested = {num_smiles}, Num. dups = {v - len(valid_smiles)}')

        # sub-nodes of sim data resource
        smiles_node = DataNode(label="smiles", data=valid_smiles)

        # add sim data nodes to parent node
        if sim_data_node:
            sim_data_node.data = [smiles_node]

        duration = time.time() - start
        print('\nModel evaluation duration: {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))

    @staticmethod
    def save_model(model, path, name):
        os.makedirs(path, exist_ok=True)
        file = os.path.join(path, name + ".mod")
        torch.save(model.state_dict(), file)

    @staticmethod
    def load_model(path, name):
        return torch.load(os.path.join(path, name), map_location=torch.device(device))


def main(flags):
    sim_label = 'GPMT-pretraining-Stack-RNN'
    if flags.eval:
        sim_label += '_eval'
    sim_data = DataNode(label=sim_label)
    nodes_list = []
    sim_data.data = nodes_list

    # For searching over multiple seeds
    hparam_search = None

    for seed in seeds:
        summary_writer_creator = lambda: SummaryWriter(log_dir="tb_gpmt"
                                                               "/{}_{}_{}/".format(sim_label, seed, dt.now().strftime(
            "%Y_%m_%d__%H_%M_%S")))

        # for data collection of this round of simulation.
        data_node = DataNode(label="seed_%d" % seed)
        nodes_list.append(data_node)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        print('---------------------------------------------------')
        print('Running on dataset: %s' % flags.data_file)
        print('---------------------------------------------------')

        trainer = GpmtPretrain()
        k = 1
        if flags["hparam_search"]:
            print("Hyperparameter search enabled: {}".format(flags["hparam_search_alg"]))

            # arguments to callables
            extra_init_args = {}
            extra_data_args = {"flags": flags}
            extra_train_args = {"is_hsearch": True,
                                "n_iters": 50000,
                                "tb_writer": summary_writer_creator}

            hparams_conf = get_hparam_config(flags)
            if hparam_search is None:
                search_alg = {"random_search": RandomSearch,
                              "bayopt_search": BayesianOptSearch}.get(flags["hparam_search_alg"],
                                                                      BayesianOptSearch)
                search_args = GPMinArgs(n_calls=20, random_state=seed)
                hparam_search = search_alg(hparam_config=hparams_conf,
                                           num_folds=1,
                                           initializer=trainer.initialize,
                                           data_provider=trainer.data_provider,
                                           train_fn=trainer.train,
                                           save_model_fn=trainer.save_model,
                                           alg_args=search_args,
                                           init_args=extra_init_args,
                                           data_args=extra_data_args,
                                           train_args=extra_train_args,
                                           data_node=data_node,
                                           split_label='',
                                           sim_label=sim_label,
                                           dataset_label='ChEMBL_SMILES',
                                           results_file="{}_{}_gpmt_{}.csv".format(
                                               flags["hparam_search_alg"], sim_label, date_label))

            stats = hparam_search.fit(model_dir="models", model_name='gpmt')
            print(stats)
            print("Best params = {}".format(stats.best()))
        else:
            hyper_params = default_hparams(flags)
            model, optimizer, gen_data, rnn_args = trainer.initialize(hyper_params,
                                                                      gen_data=trainer.data_provider(k, flags)['train'])
            if flags.eval:
                model.load_state_dict(trainer.load_model(flags.model_dir, flags.eval_model_name))
                trainer.evaluate_model(model, gen_data, rnn_args, data_node, num_smiles=300)
            else:
                results = trainer.train(model=model,
                                        optimizer=optimizer,
                                        gen_data=gen_data,
                                        rnn_args=rnn_args,
                                        n_iters=1500000,
                                        sim_data_node=data_node,
                                        tb_writer=summary_writer_creator)
                trainer.save_model(results['model'], flags.model_dir,
                                   name=f'gpmt-pretrained_stack-rnn_{hyper_params["unit_type"]}_'
                                        f'{date_label}_{results["score"]}_{results["epoch"]}')

    # save simulation data resource tree to file.
    sim_data.to_json(path="./analysis/")


def default_hparams(args):
    return {
        'unit_type': 'gru',
        'num_layers': 1,
        'dropout': 0.0,
        'd_model': 1500,
        'stack_width': 1500,
        'stack_depth': 200,
        'batch_size': 1,

        # optimizer params
        'optimizer': 'adadelta',
        # 'optimizer__global__weight_decay': 0.00005,
        'optimizer__global__lr': 0.001,
    }


def get_hparam_config(args):
    config = {
        'unit_type': CategoricalParam(choices=['gru', 'lstm']),
        'num_layers': DiscreteParam(min=1, max=10),
        "d_model": DiscreteParam(min=32, max=1024),
        "stack_width": DiscreteParam(min=10, max=128),
        "stack_depth": DiscreteParam(min=10, max=64),
        "dropout": RealParam(0.0, max=0.3),
        "batch_size": CategoricalParam(choices=[32, 64, 128]),

        # optimizer params
        "optimizer": CategoricalParam(choices=["sgd", "adam", "adadelta", "adagrad", "adamax", "rmsprop"]),
        "optimizer__global__weight_decay": LogRealParam(),
        "optimizer__global__lr": LogRealParam(),
    }
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pretraining of Memory-Augmented Transformer.')
    parser.add_argument('-d', '--data',
                        type=str,
                        dest='data_file',
                        help='Train data file')
    parser.add_argument('--model_dir',
                        type=str,
                        default='./model_dir',
                        help='Directory to store the log files in the training process.'
                        )
    parser.add_argument("--hparam_search",
                        action="store_true",
                        help="If true, hyperparameter searching would be performed.")
    parser.add_argument("--hparam_search_alg",
                        type=str,
                        default="bayopt_search",
                        help="Hyperparameter search algorithm to use. One of [bayopt_search, random_search]")
    parser.add_argument("--eval",
                        action="store_true",
                        help="If true, a saved model is loaded and evaluated")
    parser.add_argument("--eval_model_name",
                        default=None,
                        type=str,
                        help="The filename of the model to be loaded from the directory specified in --model_dir")

    args = parser.parse_args()
    flags = Flags()
    args_dict = args.__dict__
    for arg in args_dict:
        setattr(flags, arg, args_dict[arg])
    main(flags)
