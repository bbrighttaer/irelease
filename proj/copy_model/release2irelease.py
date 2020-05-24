# Author: bbrighttaer
# Project: GPMT
# Date: 5/19/2020
# Time: 8:28 PM
# File: release2irelease.py

from __future__ import absolute_import, division, print_function, unicode_literals

import os

import torch
import torch.nn as nn

from gpmt.data import GeneratorData
from gpmt.model import Encoder, StackRNN, StackRNNLinear
from gpmt.utils import get_default_tokens
from proj.copy_model.stackRNN import StackAugmentedRNN

hidden_size = 1500
stack_width = 1500
stack_depth = 200
layer_type = 'GRU'
lr = 0.001
optimizer_instance = torch.optim.Adadelta
use_cuda = torch.cuda.is_available()

tokens = get_default_tokens()
gen_data = GeneratorData(training_data_path='../../data/chembl_xsmall.smi',
                         delimiter='\t',
                         cols_to_read=[0],
                         keep_header=True,
                         pad_symbol=' ',
                         max_len=120,
                         tokens=tokens,
                         use_cuda=use_cuda)


def default_hparams(args):
    return {'d_model': 1500,
            'dropout': 0.0,
            'agent_params': {'unit_type': 'gru',
                             'num_layers': 1,
                             'stack_width': 1500,
                             'stack_depth': 200,
                             'optimizer': 'adadelta',
                             'optimizer__global__weight_decay': 0.00005,
                             'optimizer__global__lr': 0.001}
            }


def create_irelease_model_and_transfer(model1):
    hparams = default_hparams(None)

    # Embeddings provider
    encoder = Encoder(vocab_size=gen_data.n_characters, d_model=hparams['d_model'],
                      padding_idx=gen_data.char2idx[gen_data.pad_symbol],
                      dropout=hparams['dropout'], return_tuple=True)
    encoder.embedding.load_state_dict(model1.encoder.state_dict())

    stack_rnn = StackRNN(layer_index=1,
                         input_size=hparams['d_model'],
                         hidden_size=hparams['d_model'],
                         has_stack=True,
                         unit_type=hparams['agent_params']['unit_type'],
                         stack_width=hparams['agent_params']['stack_width'],
                         stack_depth=hparams['agent_params']['stack_depth'],
                         k_mask_func=encoder.k_padding_mask)
    stack_rnn.stack_controls_layer.load_state_dict(model1.stack_controls_layer.state_dict())
    stack_rnn.stack_input_layer.load_state_dict(model1.stack_input_layer.state_dict())
    stack_rnn.rnn.load_state_dict(model1.rnn.state_dict())

    rnn_linear_out = StackRNNLinear(out_dim=gen_data.n_characters, hidden_size=hparams['d_model'], bidirectional=False,
                                    bias=True)
    rnn_linear_out.decoder.load_state_dict(model1.decoder.state_dict())
    agent_net = nn.Sequential(encoder,
                              stack_rnn,
                              # StackedRNNDropout(hparams['dropout']),
                              # StackedRNNLayerNorm(hparams['d_model']),
                              rnn_linear_out)
    return agent_net


def save_model(model, path, name):
    os.makedirs(path, exist_ok=True)
    file = os.path.join(path, name + ".mod")
    torch.save(model.state_dict(), file)


if __name__ == '__main__':
    my_generator = StackAugmentedRNN(input_size=gen_data.n_characters,
                                     hidden_size=hidden_size,
                                     output_size=gen_data.n_characters,
                                     layer_type=layer_type,
                                     n_layers=1, is_bidirectional=False, has_stack=True,
                                     stack_width=stack_width, stack_depth=stack_depth,
                                     use_cuda=use_cuda,
                                     optimizer_instance=optimizer_instance, lr=lr)

    my_generator.load_model('prior.mod', loc='cpu')
    irelease_model = create_irelease_model_and_transfer(my_generator)
    save_model(irelease_model, '../model_dir/', 'irelease_prior')
    print('Model weights transfer completed!')
