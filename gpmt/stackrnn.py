# Author: bbrighttaer
# Project: GPMT
# Date: 3/30/2020
# Time: 12:57 PM
# File: stackrnn.py

from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn


class StackRNNCell(nn.Module):
    """
    Stack RNN implementation of https://arxiv.org/abs/1503.01007
    Credits: https://github.com/isayev/ReLeaSE
    """

    def __init__(self, input_size, hidden_size, has_stack, unit_type='lstm', bias=True, stack_depth=None,
                 stack_width=None):
        super(StackRNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.unit_type = unit_type
        self.has_stack = has_stack
        self.stack_depth = stack_depth
        self.stack_width = stack_width
        if has_stack:
            self.input_size = input_size + stack_width
            self.A_linear = nn.Linear(hidden_size, 3)
            self.D_linear = nn.Linear(hidden_size, stack_width)
        if self.unit_type == 'lstm':
            self.rnn_cell = nn.LSTMCell(self.input_size, self.hidden_size, bias)
        else:
            self.rnn_cell = nn.GRUCell(self.input_size, self.hidden_size, bias)

    def forward(self, x, h, c, stack_prev=None):
        if self.has_stack:
            assert stack_prev is not None
            stack_top = stack_prev[:, 0, :]
            x = torch.cat([x, stack_top], dim=-1)
            if self.unit_type == 'lstm':
                hidden, cell_state = self.rnn_cell(x, (h, c))
                out = hidden, cell_state
            else:
                hidden = self.rnn_cell(x, h)
                out = hidden

            # stack update
            controls = torch.softmax(self.A_linear(hidden), dim=-1)
            stack_input = torch.tanh(self.D_linear(hidden))
            stack = self.stack_augmentation(stack_input, stack_prev, controls)
        else:
            stack = None
            out = self.rnn_cell(x, h)
        return out, stack


