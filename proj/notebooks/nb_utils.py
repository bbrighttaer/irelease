# Author: bbrighttaer
# Project: irelease
# Date: 6/25/2020
# Time: 6:51 AM
# File: nb_utils.py

from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
import numpy as np

from irelease.data import GeneratorData
from irelease.model import Encoder, StackRNN, StackedRNNDropout, StackedRNNLayerNorm, StackRNNLinear, CriticRNN, \
    RewardNetRNN
from irelease.mol_metrics import verify_sequence, get_mol_metrics
from irelease.predictor import RNNPredictor
from irelease.utils import get_default_tokens, init_hidden, init_cell, init_stack

if torch.cuda.is_available():
    dvc_id = 2
    use_cuda = True
    device = f'cuda:{dvc_id}'
    torch.cuda.set_device(dvc_id)
else:
    device = 'cpu'
    use_cuda = None

__all__ = ['agent_net_hidden_states_func', 'data_provider', 'initialize', 'evaluate', 'device', 'use_cuda']


def agent_net_hidden_states_func(batch_size, num_layers, hidden_size, stack_depth, stack_width, unit_type):
    return [get_initial_states(batch_size, hidden_size, 1, stack_depth, stack_width, unit_type) for _ in
            range(num_layers)]


def get_initial_states(batch_size, hidden_size, num_layers, stack_depth, stack_width, unit_type):
    hidden = init_hidden(num_layers=num_layers, batch_size=batch_size, hidden_size=hidden_size, num_dir=1,
                         dvc=device)
    if unit_type == 'lstm':
        cell = init_cell(num_layers=num_layers, batch_size=batch_size, hidden_size=hidden_size, num_dir=1,
                         dvc=device)
    else:
        cell = None
    stack = init_stack(batch_size, stack_width, stack_depth, dvc=device)
    return hidden, cell, stack


def initialize(hparams, demo_data_gen, unbiased_data_gen, has_critic):
    # Embeddings provider
    encoder = Encoder(vocab_size=demo_data_gen.n_characters, d_model=hparams['d_model'],
                      padding_idx=demo_data_gen.char2idx[demo_data_gen.pad_symbol],
                      dropout=hparams['dropout'], return_tuple=True)

    # Agent entities
    rnn_layers = []
    has_stack = True
    for i in range(1, hparams['agent_params']['num_layers'] + 1):
        rnn_layers.append(StackRNN(layer_index=i,
                                   input_size=hparams['d_model'],
                                   hidden_size=hparams['d_model'],
                                   has_stack=has_stack,
                                   unit_type=hparams['agent_params']['unit_type'],
                                   stack_width=hparams['agent_params']['stack_width'],
                                   stack_depth=hparams['agent_params']['stack_depth'],
                                   k_mask_func=encoder.k_padding_mask))
        if hparams['agent_params']['num_layers'] > 1:
            rnn_layers.append(StackedRNNDropout(hparams['dropout']))
            rnn_layers.append(StackedRNNLayerNorm(hparams['d_model']))
    agent_net = nn.Sequential(encoder,
                              *rnn_layers,
                              StackRNNLinear(out_dim=demo_data_gen.n_characters,
                                             hidden_size=hparams['d_model'],
                                             bidirectional=False,
                                             bias=True))
    agent_net = agent_net.to(device)
    init_state_args = {'num_layers': hparams['agent_params']['num_layers'],
                       'hidden_size': hparams['d_model'],
                       'stack_depth': hparams['agent_params']['stack_depth'],
                       'stack_width': hparams['agent_params']['stack_width'],
                       'unit_type': hparams['agent_params']['unit_type']}
    if has_critic:
        critic = nn.Sequential(encoder,
                               CriticRNN(hparams['d_model'], hparams['critic_params']['d_model'],
                                         unit_type=hparams['critic_params']['unit_type'],
                                         dropout=hparams['critic_params']['dropout'],
                                         num_layers=hparams['critic_params']['num_layers']))
        critic = critic.to(device)
    else:
        critic = None

    # Reward function entities
    reward_net = nn.Sequential(encoder,
                               RewardNetRNN(input_size=hparams['d_model'],
                                            hidden_size=hparams['reward_params']['d_model'],
                                            num_layers=hparams['reward_params']['num_layers'],
                                            bidirectional=hparams['reward_params']['bidirectional'],
                                            use_attention=hparams['reward_params']['use_attention'],
                                            dropout=hparams['reward_params']['dropout'],
                                            unit_type=hparams['reward_params']['unit_type'],
                                            use_smiles_validity_flag=hparams['reward_params']['use_validity_flag']))
    reward_net = reward_net.to(device)
    expert_model = RNNPredictor(hparams['expert_model_params'], device)
    demo_data_gen.set_batch_size(hparams['reward_params']['demo_batch_size'])

    init_args = {'agent_net': agent_net,
                 'critic_net': critic,
                 'reward_net': reward_net,
                 'gamma': hparams['gamma'],
                 'expert_model': expert_model,
                 'demo_data_gen': demo_data_gen,
                 'unbiased_data_gen': unbiased_data_gen,
                 'init_hidden_states_args': init_state_args,
                 'gen_args': {'num_layers': hparams['agent_params']['num_layers'],
                              'hidden_size': hparams['d_model'],
                              'num_dir': 1,
                              'stack_depth': hparams['agent_params']['stack_depth'],
                              'stack_width': hparams['agent_params']['stack_width'],
                              'has_stack': has_stack,
                              'has_cell': hparams['agent_params']['unit_type'] == 'lstm',
                              'device': device}}
    return init_args


def data_provider(demo_file, unbiased_file):
    tokens = get_default_tokens()
    demo_data = GeneratorData(training_data_path=demo_file,
                              delimiter='\t',
                              cols_to_read=[0],
                              keep_header=True,
                              pad_symbol=' ',
                              max_len=120,
                              tokens=tokens,
                              use_cuda=use_cuda)
    unbiased_data = GeneratorData(training_data_path=unbiased_file,
                                  delimiter='\t',
                                  cols_to_read=[0],
                                  keep_header=True,
                                  pad_symbol=' ',
                                  max_len=120,
                                  tokens=tokens,
                                  use_cuda=use_cuda)
    return {'demo_data': demo_data, 'unbiased_data': unbiased_data}


def evaluate(generated_smiles, ref_smiles):
    res_dict = {}
    smiles = []
    for s in generated_smiles:
        if verify_sequence(s):
            smiles.append(s)
    mol_metrics = get_mol_metrics()
    for metric in mol_metrics:
        res_dict[metric] = np.mean(mol_metrics[metric](smiles, ref_smiles))
    return res_dict
