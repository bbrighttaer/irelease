# Author: bbrighttaer
# Project: irelease
# Date: 6/25/2020
# Time: 6:51 AM
# File: nb_utils.py

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import json
import torch
import torch.nn as nn
import numpy as np

from irelease.data import GeneratorData
from irelease.model import Encoder, StackRNN, StackedRNNDropout, StackedRNNLayerNorm, RNNLinearOut, CriticRNN, \
    RewardNetRNN
from irelease.mol_metrics import verify_sequence, get_mol_metrics
from irelease.predictor import RNNPredictor
from irelease.utils import get_default_tokens, init_hidden, init_cell, init_stack, canonical_smiles, seq2tensor, \
    pad_sequences, ExpAverage

if torch.cuda.is_available():
    dvc_id = 0
    use_cuda = True
    device = f'cuda:{dvc_id}'
    torch.cuda.set_device(dvc_id)
else:
    device = 'cpu'
    use_cuda = None


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
                      dropout=hparams['dropout'], return_tuple=True).eval()

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
                              RNNLinearOut(out_dim=demo_data_gen.n_characters,
                                           hidden_size=hparams['d_model'],
                                           bidirectional=False,
                                           bias=True))
    agent_net = agent_net.to(device).eval()
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
        critic = critic.to(device).eval()
    else:
        critic = None

    # Reward function entities
    reward_net_rnn = RewardNetRNN(input_size=hparams['d_model'], hidden_size=hparams['reward_params']['d_model'],
                                  num_layers=hparams['reward_params']['num_layers'],
                                  bidirectional=hparams['reward_params']['bidirectional'],
                                  use_attention=hparams['reward_params']['use_attention'],
                                  dropout=hparams['reward_params']['dropout'],
                                  unit_type=hparams['reward_params']['unit_type'],
                                  use_smiles_validity_flag=hparams['reward_params']['use_validity_flag'])
    reward_net = nn.Sequential(encoder,
                               reward_net_rnn)
    reward_net = reward_net.to(device)
    # expert_model = RNNPredictor(hparams['expert_model_params'], device)
    demo_data_gen.set_batch_size(hparams['reward_params']['demo_batch_size'])

    init_args = {'agent_net': agent_net,
                 'critic_net': critic,
                 'reward_net': reward_net,
                 'reward_net_rnn': reward_net_rnn,
                 'encoder': encoder.eval(),
                 'gamma': hparams['gamma'],
                 # 'expert_model': expert_model,
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


def load_model_weights(path):
    return torch.load(path, map_location=torch.device(device))


def logp_ppo_hparams():
    return {'d_model': 1500,
            'dropout': 0.0,
            'monte_carlo_N': 5,
            'use_monte_carlo_sim': True,
            'no_mc_fill_val': 0.0,
            'gamma': 0.97,
            'episodes_to_train': 10,
            'gae_lambda': 0.95,
            'ppo_eps': 0.2,
            'ppo_batch': 1,
            'ppo_epochs': 5,
            'entropy_beta': 0.01,
            'use_true_reward': False,
            'reward_params': {'num_layers': 2,
                              'd_model': 512,
                              'unit_type': 'gru',
                              'demo_batch_size': 32,
                              'irl_alg_num_iter': 5,
                              'dropout': 0.2,
                              'use_attention': False,
                              'use_validity_flag': True,
                              'bidirectional': True,
                              'optimizer': 'adadelta',
                              'optimizer__global__weight_decay': 0.0005,
                              'optimizer__global__lr': 0.001, },
            'agent_params': {'unit_type': 'gru',
                             'num_layers': 2,
                             'stack_width': 1500,
                             'stack_depth': 200,
                             'optimizer': 'adadelta',
                             'optimizer__global__weight_decay': 0.0000,
                             'optimizer__global__lr': 0.001},
            'critic_params': {'num_layers': 2,
                              'd_model': 256,
                              'dropout': 0.2,
                              'unit_type': 'lstm',
                              'optimizer': 'adadelta',
                              'optimizer__global__weight_decay': 0.00005,
                              'optimizer__global__lr': 0.001},
            'expert_model_params': {'model_dir': '../model_dir/expert_rnn_reg',
                                    'd_model': 128,
                                    'rnn_num_layers': 2,
                                    'dropout': 0.8,
                                    'is_bidirectional': False,
                                    'unit_type': 'lstm'}
            }


def logp_reinforce_hparams():
    return {'d_model': 1500,
            'dropout': 0.0,
            'monte_carlo_N': 5,
            'use_monte_carlo_sim': False,
            'no_mc_fill_val': 0.0,
            'gamma': 0.97,
            'episodes_to_train': 10,
            'reinforce_max_norm': None,
            'lr_decay_gamma': 0.1,
            'lr_decay_step_size': 1000,
            'xent_lambda': 0.0,
            'use_true_reward': False,
            'reward_params': {'num_layers': 2,
                              'd_model': 256,
                              'unit_type': 'lstm',
                              'demo_batch_size': 32,
                              'irl_alg_num_iter': 5,
                              'dropout': 0.0,
                              'use_attention': False,
                              'use_validity_flag': True,
                              'bidirectional': True,
                              'optimizer': 'adadelta',
                              'optimizer__global__weight_decay': 0.0000,
                              'optimizer__global__lr': 0.001, },
            'agent_params': {'unit_type': 'gru',
                             'num_layers': 2,
                             'stack_width': 1500,
                             'stack_depth': 200,
                             'optimizer': 'adadelta',
                             'optimizer__global__weight_decay': 0.0000,
                             'optimizer__global__lr': 0.001},
            'expert_model_params': {'model_dir': '../model_dir/expert_rnn_reg',
                                    'd_model': 128,
                                    'rnn_num_layers': 2,
                                    'dropout': 0.8,
                                    'is_bidirectional': False,
                                    'unit_type': 'lstm'}
            }


def smiles_to_tensor(smiles):
    smiles = list(smiles)
    _, valid_vec = canonical_smiles(smiles)
    valid_vec = torch.tensor(valid_vec).view(-1, 1).float().to(device)
    smiles, _ = pad_sequences(smiles)
    inp, _ = seq2tensor(smiles, tokens=get_default_tokens())
    inp = torch.from_numpy(inp).long().to(device)
    return inp, valid_vec


def smoothing_values(values, beta=0.9):
    exp_avg = ExpAverage(beta)
    smooth_v = []
    for v in values:
        exp_avg.update(v)
        smooth_v.append(exp_avg.value)
    return smooth_v


def get_convergence_data(file):
    with open(file, 'r') as f:
        train_hist = json.load(f)
    lbl = file.split('/')[-1].split('.')[0]
    baseline_mean_vals = train_hist[lbl][0][list(train_hist[lbl][0].keys())[0]][0]['baseline_mean_vals']
    demo_mean_vals = train_hist[lbl][0][list(train_hist[lbl][0].keys())[0]][1]['biased_mean_vals']
    biased_mean_vals = train_hist[lbl][0][list(train_hist[lbl][0].keys())[0]][2]['gen_mean_vals']
    return {'baseline': baseline_mean_vals, 'demo': demo_mean_vals, 'biased': biased_mean_vals}


def smiles_from_json_data(file):
    valid_smiles = []
    invalid_smiles = []
    if os.path.exists(file):
        with open(file, 'r') as f:
            data = json.load(f)
        for k in data:
            if k == 'metadata':
                continue
            if data[k]:
                for seed_data in data[k]:
                    for gen in seed_data:
                        valid_smiles.extend(seed_data[gen][0]['valid_smiles'])
                        invalid_smiles.extend(seed_data[gen][1]['invalid_smiles'])
    return valid_smiles, invalid_smiles

# if __name__ == '__main__':
#     get_convergence_data('./jak2_min/JAK2_min_Stack_RNN_XEnt_Generator_Baseline.json')
