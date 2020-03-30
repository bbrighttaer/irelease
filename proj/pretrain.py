# Author: bbrighttaer
# Project: GPMT
# Date: 3/23/2020
# Time: 12:03 PM
# File: pretrain.py

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import argparse
import math
import time
import copy
from datetime import datetime as dt
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as sch
from sklearn.metrics import accuracy_score
from soek import CategoricalParam, LogRealParam, ConstantParam, RealParam, DiscreteParam, DataNode, RandomSearch, \
    BayesianOptSearch
from soek.bopt import GPMinArgs, GBRTMinArgs
from soek.template import Trainer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from gpmt.data import GeneratorData
from gpmt.model import StackDecoderLayer, Encoder, PositionalEncoding, AttentionInitialize, AttentionTerminal, \
    NonsatActivation, AttentionOptimizer, LinearOut, LabelSmoothing, get_std_opt
from gpmt.tboard import TBMeanTracker
from gpmt.utils import Flags, get_default_tokens, parse_optimizer, ExpAverage, GradStats, Count

device = 'cuda'

currentDT = dt.now()
date_label = currentDT.strftime("%Y_%m_%d__%H_%M_%S")

seeds = [1]

if torch.cuda.is_available():
    dvc_id = 2
    use_cuda = True
    torch.cuda.set_device(dvc_id)
else:
    use_cuda = None
    dvc_id = 0


class GpmtPretrain(Trainer):
    @staticmethod
    def initialize(hparams, gen_data, *args, **kwargs):
        gen_data.set_batch_size(hparams['batch_size'])

        # Create stack-augmented transformer (Decoder) layer(s)
        encoder = Encoder(vocab_size=gen_data.n_characters, d_model=hparams['d_model'],
                          padding_idx=gen_data.char2idx[gen_data.pad_symbol],
                          dropout=hparams['dropout'])
        attn_layers = []
        for i in range(hparams['attn_layers']):
            attn_layers.append(
                StackDecoderLayer(d_model=hparams['d_model'],
                                  num_heads=hparams['attn_heads'],
                                  d_hidden=hparams['d_hidden'],
                                  stack_depth=hparams['stack_depth'],
                                  stack_width=hparams['stack_width'],
                                  d_ff=hparams['d_ff'],
                                  d_ss=hparams['d_ss'],
                                  dropout=hparams['dropout'],
                                  k_mask_func=encoder.k_padding_mask,
                                  use_memory=True)
            )

        # Create classifier layers (post-attention layers)
        classifier_layers = []
        p = hparams['d_model']
        for dim in hparams['lin_dims']:
            classifier_layers.append(nn.Linear(p, dim))
            classifier_layers.append(nn.LayerNorm(dim))
            classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Dropout(hparams['dropout']))
            p = dim
        classifier_layers.append(nn.Linear(p, gen_data.n_characters))
        # classifier_layers.append(LinearOut(encoder.embeddings_weight, p, hparams['d_model'], hparams['dropout']))

        # Create main model
        model = nn.Sequential(encoder,
                              PositionalEncoding(d_model=hparams['d_model'],
                                                 dropout=hparams['dropout']),
                              AttentionInitialize(d_hidden=hparams['d_hidden'],
                                                  s_width=hparams['stack_width'],
                                                  s_depth=hparams['stack_depth'],
                                                  dvc=f'{device}:{dvc_id}'),
                              *attn_layers,
                              AttentionTerminal(),
                              *classifier_layers)
        if use_cuda:
            model = model.cuda()

        optimizer = parse_optimizer(hparams, model)
        # optimizer = get_std_opt(model, hparams['d_model'])
        # optimizer = AttentionOptimizer(model_size=hparams['d_model'],
        #                                factor=2,
        #                                warmup=4000,
        #                                optimizer=parse_optimizer(hparams, model))

        return model, optimizer, gen_data

    @staticmethod
    def data_provider(k, flags):
        gen_data = GeneratorData(training_data_path=flags.data_file,
                                 delimiter='\t',
                                 cols_to_read=[0],
                                 keep_header=True,
                                 pad_symbol=' ',
                                 max_len=1000,
                                 tokens=None,
                                 use_cuda=use_cuda,
                                 tokens_reload=True)
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
    def train(model, optimizer, gen_data, n_iters=5000, sim_data_node=None, epoch_ckpt=(2, 1.0), tb_writer=None,
              is_hsearch=False):
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
        # criterion = LabelSmoothing(gen_data.n_characters, gen_data.char2idx[gen_data.pad_symbol], 0.1)

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
            for epoch in range(n_epochs):
                if terminate_training:
                    print("Terminating training...")
                    break
                for phase in ["train", "val" if is_hsearch else "test"]:
                    if phase == "train":
                        print("Training....")
                        # Training mode
                        model.train()
                    else:
                        print("Validation...")
                        # Evaluation mode
                        model.eval()

                    epoch_losses = []
                    epoch_scores = []

                    # Iterate through mini-batches
                    # with TBMeanTracker(tb_writer, 10) as tracker:
                    with grad_stats:
                        for b in tqdm(range(num_batches)):
                            inputs, labels = gen_data.random_training_set()

                            optimizer.zero_grad()

                            # track history if only in train
                            with torch.set_grad_enabled(phase == "train"):
                                # forward propagation
                                predictions = model(inputs)
                                predictions = predictions.permute(1, 0, -1)
                                predictions = predictions.contiguous().view(-1, predictions.shape[-1])
                                labels = labels.contiguous().view(-1)

                                # calculate loss
                                loss = criterion(predictions, labels)

                            # fail fast
                            if str(loss.item()) == "nan":
                                terminate_training = True
                                break

                            # metrics
                            eval_dict = {}
                            score = GpmtPretrain.evaluate(eval_dict, predictions, labels)

                            # TBoard info
                            # tracker.track("%s/loss" % phase, loss.item(), tb_idx[phase].IncAndGet())
                            # tracker.track("%s/score" % phase, score, tb_idx[phase].i)
                            # for k in eval_dict:
                            #     tracker.track('{}/{}'.format(phase, k), eval_dict[k], tb_idx[phase].i)

                            if phase == "train":
                                # backward pass
                                loss.backward()
                                optimizer.step()

                                # for epoch stats
                                epoch_losses.append(loss.item())

                                # for sim data resource
                                train_scores_lst.append(score)
                                loss_lst.append(loss.item())

                                print("\tEpoch={}/{}, batch={}/{}, "
                                      "pred_loss={:.4f}, accuracy: {:.2f}".format(epoch + 1, n_epochs,
                                                                                  b + 1,
                                                                                  num_batches,
                                                                                  loss.item(), eval_dict['accuracy']))
                            else:
                                # for epoch stats
                                epoch_scores.append(score)

                                # for sim data resource
                                scores_lst.append(score)
                                for m in eval_dict:
                                    if m in metrics_dict:
                                        metrics_dict[m].append(eval_dict[m])
                                    else:
                                        metrics_dict[m] = [eval_dict[m]]

                                print("\nEpoch={}/{}, batch={}/{}, "
                                      "evaluation results= {}, accuracy={}".format(epoch + 1, n_epochs, b + 1,
                                                                                   num_batches, eval_dict, score))
                    # End of mini=batch iterations.

                    if phase == "train":
                        ep_loss = np.nanmean(epoch_losses)
                        e_avg.update(ep_loss)
                        if epoch % (epoch_ckpt[0] - 1) == 0 and epoch > 0:
                            if e_avg.value > epoch_ckpt[1]:
                                terminate_training = True
                        print("\nPhase: {}, avg task pred_loss={:.4f}, ".format(phase, np.nanmean(epoch_losses)))
                        # scheduler.step()
                    else:
                        mean_score = np.mean(epoch_scores)
                        if best_score < mean_score:
                            best_score = mean_score
                            best_model_wts = copy.deepcopy(model.state_dict())
                            best_epoch = epoch
        except RuntimeError as e:
            print(str(e))

        duration = time.time() - start
        print('\nModel training duration: {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))
        try:
            model.load_state_dict(best_model_wts)
        except RuntimeError as e:
            print(str(e))
        return {'model': model, 'score': best_score, 'epoch': best_epoch}

    @staticmethod
    def evaluate_model(*args, **kwargs):
        super().evaluate_model(*args, **kwargs)

    @staticmethod
    def save_model(model, path, name):
        os.makedirs(path, exist_ok=True)
        file = os.path.join(path, name + ".mod")
        # torch.save(model.state_dict(), file)

    @staticmethod
    def load_model(path, name):
        # if dvc is None:
        #     dvc = torch.device("cuda:0")
        return torch.load(os.path.join(path, name),
                          map_location=torch.device(device))


def main(flags):
    sim_label = 'GPMT-pretraining-memory'
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

        # load data
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
            hyper_params = default_hparams_bopt(flags)
            model, optimizer, gen_data = trainer.initialize(hyper_params,
                                                            gen_data=trainer.data_provider(k, flags)['train'])
            results = trainer.train(model=model,
                                    optimizer=optimizer,
                                    gen_data=gen_data,
                                    n_iters=500000,
                                    sim_data_node=data_node,
                                    tb_writer=summary_writer_creator)
            trainer.save_model(results['model'], flags.model_dir,
                               name=f'gpmt-pretrained_{date_label}_{results["score"]}_{results["epoch"]}')

    # save simulation data resource tree to file.
    sim_data.to_json(path="./analysis/")


def default_hparams_bopt(args):
    return {
        'attn_heads': 8,
        'attn_layers': 3,
        'lin_dims': [1024, 256],
        'dropout': 0.2,
        'd_model': 128,
        'd_hidden': 128,
        'stack_width': 64,
        'stack_depth': 100,
        'd_ff': 2048,
        "d_ss": 512,
        'batch_size': 32,

        # optimizer params
        'optimizer': 'adam',
        'optimizer__global__weight_decay': 0.00005,
        'optimizer__global__lr': 0.001,
    }


def get_hparam_config(args):
    config = {
        "attn_heads": CategoricalParam([1, 2, 4, 8]),
        "attn_layers": DiscreteParam(min=1, max=4),
        "lin_dims": DiscreteParam(min=64, max=2048, size=DiscreteParam(min=1, max=3)),
        "d_model": CategoricalParam(choices=[128, 256, 512, 1024]),
        "d_hidden": DiscreteParam(min=10, max=64),
        "stack_width": DiscreteParam(min=10, max=64),
        "stack_depth": DiscreteParam(min=10, max=64),
        "d_ff": DiscreteParam(min=128, max=2048),
        "d_ss": DiscreteParam(min=128, max=2024),
        "dropout": RealParam(0.0, max=0.5),
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
