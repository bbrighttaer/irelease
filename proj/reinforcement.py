# Author: bbrighttaer
# Project: GPMT
# Date: 4/8/2020
# Time: 8:02 PM
# File: reinforcement.py

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
from soek import Trainer, DataNode
from torch.utils.tensorboard import SummaryWriter

from gpmt.data import GeneratorData
from gpmt.model import Encoder, RewardNetRNN
from gpmt.utils import Flags, get_default_tokens, parse_optimizer

currentDT = dt.now()
date_label = currentDT.strftime("%Y_%m_%d__%H_%M_%S")

seeds = [1]

if torch.cuda.is_available():
    dvc_id = 3
    use_cuda = True
    device = 'cuda'
    torch.cuda.set_device(dvc_id)
else:
    device = 'cpu'
    use_cuda = None
    dvc_id = 0


class IReLeaSE(Trainer):

    @staticmethod
    def initialize(hparams, gen_data, *args, **kwargs):
        encoder = Encoder(vocab_size=gen_data.n_characters, d_model=hparams['d_model'],
                          padding_idx=gen_data.char2idx[gen_data.pad_symbol],
                          dropout=hparams['dropout'], return_tuple=True)
        reward_net = RewardNetRNN(input_size=hparams['d_model'],
                                  hidden_size=hparams['hidden_size'],
                                  num_layers=hparams['rw_net_num_layers'],
                                  bidirectional=True,
                                  dropout=hparams['dropout'],
                                  unit_type=hparams['rw_net_unit_type'])


        optimizer = parse_optimizer(hparams, model)
        init_args = {'device': f'{device}:{dvc_id}',
                     'batch_size': hparams['batch_size'],
                     'gen_demo_data': gen_data}
        return init_args

    @staticmethod
    def data_provider(k, flags):
        tokens = get_default_tokens()
        gen_data = GeneratorData(training_data_path=flags.demo_file,
                                 delimiter='\t',
                                 cols_to_read=[0],
                                 keep_header=True,
                                 pad_symbol=' ',
                                 max_len=120,
                                 tokens=tokens,
                                 use_cuda=use_cuda)
        return {"train": gen_data, "val": gen_data, "test": gen_data}

    @staticmethod
    def evaluate(*args, **kwargs):
        super().evaluate(*args, **kwargs)

    @staticmethod
    def train(*args, **kwargs):
        super().train(*args, **kwargs)

    @staticmethod
    def evaluate_model(*args, **kwargs):
        super().evaluate_model(*args, **kwargs)

    @staticmethod
    def save_model(model, path, name):
        os.makedirs(path, exist_ok=True)
        file = os.path.join(path, name + ".mod")
        torch.save(model.state_dict(), file)

    @staticmethod
    def load_model(path, name):
        return torch.load(os.path.join(path, name), map_location=torch.device(f'{device}:{dvc_id}'))


def main(flags):
    sim_label = 'DeNovo-IReLeaSE'
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
        print('Demonstrations file: %s' % flags.demo_file)
        print('---------------------------------------------------')

        irelease = IReLeaSE()
        k = 1
        if flags.hparam_search:
            pass
        else:
            hyper_params = default_hparams(flags)


    # save simulation data resource tree to file.
    sim_data.to_json(path="./analysis/")


def default_hparams(args):
    return {}


def get_hparam_config(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IRL for Structural Evolution of Small Molecules')
    parser.add_argument('-d', '--demo', dest='demo_file', type=str,
                        help='File containing SMILES strings which are demonstrations of the required objective')
    parser.add_argument("--hparam_search",
                        action="store_true",
                        help="If true, hyperparameter searching would be performed.")
    parser.add_argument("--hparam_search_alg",
                        type=str,
                        default="bayopt_search",
                        help="Hyperparameter search algorithm to use. One of [bayopt_search, random_search]")

    args = parser.parse_args()
    flags = Flags()
    args_dict = args.__dict__
    for arg in args_dict:
        setattr(flags, arg, args_dict[arg])
    main(flags)
