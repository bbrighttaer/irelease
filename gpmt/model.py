# Author: bbrighttaer
# Project: GPMT
# Date: 3/22/2020
# Time: 9:13 AM
# File: model.py

from __future__ import absolute_import, division, print_function, unicode_literals

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from gpmt.stackrnn import StackRNNCell
from gpmt.utils import init_hidden_2d, init_stack_2d, init_hidden, init_cell, init_stack


def clone(module, N):
    """
    Make N copies of an nn.Module

    :param module: the model to copy
    :param N: the number of copies
    :return: an nn.ModuleList of the copies
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, padding_idx, dropout=0.):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.padding_idx = padding_idx
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.k_mask = None

    def k_padding_mask(self):
        return self.k_mask

    def embeddings_weight(self):
        return self.embedding.weight

    def forward(self, x):
        """
        Retrieve embeddings for x using the indices.
        :param x: tensor
            x.shape = (batch_size, sequence length)
        :return: tensor
            x of shape (sequence length, batch_size, d_model)
        """
        self.k_mask = x == self.padding_idx
        x = self.dropout(self.embedding(x) * math.sqrt(self.d_model))
        x = x.permute(1, 0, 2)
        return x


class PositionalEncoding(nn.Module):
    """
    Implement the Positional Encoding (PE) function.
    Original source of PE: http://nlp.seas.harvard.edu/2018/04/03/attention.html
    """

    def __init__(self, d_model, dropout=0., max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.linear = nn.Linear(d_model * 2, d_model)
        self.normalize = nn.LayerNorm(d_model)
        # self.nonlinearity = NonsatActivation()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        position = position.float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Concatenates the positional encodings to the input.
        Assumes x is organized as: (Length, batch size, d_model)
        :param x:
        :return:
            x of shape (Length, batch size, d_model)
        """
        seq_length, batch_size, _ = x.shape
        x_pe = self.pe[:seq_length, :]
        x_pe = torch.repeat_interleave(x_pe.unsqueeze(1), batch_size, dim=1)
        x = torch.cat([x, x_pe], dim=-1)
        x = self.dropout(torch.relu(self.normalize(self.linear(x))))
        return x


class NonsatActivation(nn.Module):
    def __init__(self, ep=1e-4, max_iter=100):
        super(NonsatActivation, self).__init__()
        self.ep = ep
        self.max_iter = max_iter

    def forward(self, x):
        return nonsat_activation(x, self.ep, self.max_iter)


def nonsat_activation(x, ep=1e-4, max_iter=100):
    """
    Implementation of the Non-saturating non-linearity described in http://proceedings.mlr.press/v28/andrew13.html

    :param x: float, tensor
        Function input
    :param ep:float, optional
        Stop condition reference point.
    :param max_iter: int, optional,
        Helps to avoid infinite iterations.
    :return:
    """
    y = x.detach().clone()
    i = 0
    while True:
        y_ = (2. * y ** 3. / 3. + x) / (y ** 2. + 1.)
        if torch.mean(torch.abs(y_ - y)) <= ep or i > max_iter:
            return y_
        else:
            i += 1
            y = y_.detach()


def _create_attn_mask(size, dvc):
    attn_mask = torch.zeros(size, size).to(dvc)
    attn_mask = attn_mask + float('-inf')
    attn_mask = torch.triu(attn_mask, diagonal=1)
    return attn_mask


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        x = x + self.dropout(sublayer(self.norm(x)))
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.lin_inner = nn.Linear(d_model, d_ff)
        self.lin_outer = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.lin_outer(self.dropout(torch.relu(self.lin_inner(x))))
        return x


class StackDecoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, d_hidden, stack_depth, stack_width, k_mask_func, d_ff=2048, d_ss=128,
                 dropout=0., use_memory=True):
        super(StackDecoderLayer, self).__init__()
        assert callable(k_mask_func), 'k_mask_func argument should be a method of the ' \
                                      'Encoder that provides the key padding mask.'
        self.use_memory = use_memory
        self.d_hidden = d_hidden
        self.stack_depth = stack_depth
        self.stack_width = stack_width
        self.k_padding_mask_func = k_mask_func
        self.multihead_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)
        self.feedforwad = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.subalyers = nn.ModuleList([SublayerConnection(d_model, dropout),
                                        SublayerConnection(d_model, dropout)])

        # stack & hidden state elements
        self.nonlinearity = NonsatActivation()
        self.W = nn.Linear(d_model, d_hidden)
        # self.R = nn.Linear(d_hidden, d_hidden)
        self.P = nn.Linear(stack_width, d_hidden)
        self.V = nn.Linear(d_model + d_hidden, d_ss, bias=False)
        self.U = nn.Linear(d_ss, d_model, bias=False)
        self.A = nn.Linear(self.d_hidden, 3)
        self.D = nn.Linear(d_hidden, stack_width)

    def forward(self, inp):
        """
        Performs a forward pass through the stack-augmented Transformer-Decoder.

        :param inp: a tuple of tensors
            Elements are:
            [0]: the input (seq. length, batch_size, d_model)
            [1]: the hidden state of the previous layer containing information about each sample.
                (batch_size, seq. length, d_hidden)
            [2]: the previous stack. (batch_size, seq. length, stack_depth, stack_width).
        :return: tuple of tensors
            [0]: the transformed input (seq. length, batch_size, d_model)
            [1]: the hidden state of the current layer containing information about each sample.
                (batch_size, seq. length, d_hidden)
            [2]: the updated stack. (batch_size, seq. length, stack_depth, stack_width).
        """
        x_in, stack_prev = inp
        seq_length, batch_size, _ = x_in.shape
        mask = _create_attn_mask(seq_length, x_in.device)
        k_mask = self.k_padding_mask_func()

        # masked multihead self-attention
        x = self.subalyers[0](x_in, lambda x: self.multihead_attention(x, x, x, key_padding_mask=k_mask,
                                                                       attn_mask=mask, need_weights=False)[0])

        stack = stack_prev
        if self.use_memory:
            # hidden state ops
            stack_x = stack_prev[:, :, 0, :].view(batch_size, seq_length, -1)
            hidden = self.W(x.permute(1, 0, 2)) + self.P(stack_x)
            hidden = self.nonlinearity(hidden)

            # stack ops
            vx = torch.cat([x, hidden.permute(1, 0, 2)], dim=-1)
            y = self.U(torch.softmax(self.V(vx), dim=-1))

            # pooling
            x = torch.stack([x, y])
            x = torch.max(x, dim=0)[0]

            # stack update
            stack_inp = self.nonlinearity(self.D(hidden)).unsqueeze(2)
            stack_controls = torch.softmax(self.A(hidden), dim=-1)
            stack = self.stack_augmentation(stack_inp, stack_prev, stack_controls)

        # FFN module
        x = self.subalyers[1](x, self.feedforwad)
        return x, stack

    def stack_augmentation(self, input_val, stack_prev, controls):
        """
        Perform stack update for the layer.

        Arguments
        ---------
        :param input_val: tensor (seq. length, batch_size, d_model)
            Input from the multihead attention module.
        :param stack_prev: tensor (batch_size, seq. length, stack_depth, stack_width)
            Previous stack.
        :param controls: tensor (batch_size, seq. length, 3, 1)
            Predicted probabilities for each operation of the stack (PUSH, POP, NO OP).
            See  https://arxiv.org/abs/1503.01007
        :return: tensor
            New stack of shape (batch_size, seq. length, stack_depth, stack_width)
        """
        batch_size, seq_length = stack_prev.shape[:2]
        controls = controls.view(batch_size, seq_length, 3, 1, 1)
        zeros_at_the_bottom = torch.zeros(batch_size, seq_length, 1, self.stack_width).to(input_val.device)
        a_push, a_pop, a_no_op = controls[:, :, 0], controls[:, :, 1], controls[:, :, 2]
        stack_down = torch.cat((stack_prev[:, :, 1:], zeros_at_the_bottom), dim=2)
        stack_up = torch.cat((input_val, stack_prev[:, :, :-1]), dim=2)
        new_stack = a_no_op * stack_prev + a_push * stack_up + a_pop * stack_down
        return new_stack


class AttentionInitialize(nn.Module):
    """Prepares the encoded input for propagation through the memory layer(s)."""

    def __init__(self, d_hidden, s_depth, s_width, dvc='cpu'):
        super(AttentionInitialize, self).__init__()
        self.d_hidden = d_hidden
        self.s_depth = s_depth
        self.s_width = s_width
        self.dvc = dvc

    def forward(self, x):
        """

        :param x: tensor
            Encoded input of shape (Seq. length, batch_size, d_model)
        :return:
        """
        # h0 = init_hidden(x.shape[1], x.shape[0], self.d_hidden, dvc=self.dvc)
        s0 = init_stack(x.shape[1], x.shape[0], self.s_depth, self.s_width, dvc=self.dvc)
        return x, s0


class AttentionTerminal(nn.Module):
    def forward(self, inp):
        """Prepares the final attention output before applying feeding to classifier."""
        return inp[0]


class LinearOut(nn.Module):
    def __init__(self, in_embed_func, in_dim, d_model, dropout=0.):
        super(LinearOut, self).__init__()
        assert callable(in_embed_func)
        self.weight_func = in_embed_func
        self.vocab_size = self.weight_func().shape[0]
        self.linear = nn.Linear(in_dim, d_model)
        self.normalize = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        U = self.weight_func()
        x = self.dropout(torch.relu(self.normalize(self.linear(x))))
        x = F.linear(x, U)
        return x


class AttentionOptimizer:
    """Credits: http://nlp.seas.harvard.edu/2018/04/03/attention.html"""

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()


def get_std_opt(model, d_model):
    return AttentionOptimizer(d_model, 2, 4000,
                              torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class LabelSmoothing(nn.Module):

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        loss = self.criterion(x, true_dist)
        return loss


class StackRNN(nn.Module):
    def __init__(self, input_size, hidden_size, has_stack, unit_type='lstm', num_layers=1, stack_width=None,
                 stack_depth=None, bias=True, dropout=0., k_mask_func=None):
        super(StackRNN, self).__init__()
        self.num_layers = int(num_layers)
        self.hidden_size = int(hidden_size)
        self.has_stack = has_stack
        self.stack_width = int(stack_width)
        self.stack_depth = int(stack_depth)
        self.unit_type = unit_type
        self.num_dir = 1
        self.normalize_x = nn.LayerNorm(hidden_size * self.num_dir)
        self.k_padding_mask_func = k_mask_func
        if has_stack:
            self.input_size = int(input_size + stack_width)
            self.A_linear = nn.Linear(hidden_size, 3)
            self.D_linear = nn.Linear(hidden_size, stack_width)
        else:
            self.input_size = int(input_size)
        if self.unit_type == 'lstm':
            self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_layers,
                               dropout=dropout,
                               bidirectional=False,
                               bias=bias)
            self.has_cell = True
        elif self.unit_type == 'gru':
            self.rnn = nn.GRU(self.input_size, self.hidden_size, self.num_layers,
                              dropout=dropout,
                              bidirectional=False,
                              bias=bias)
            self.has_cell = False

    def forward(self, x, **kwargs):
        """
        Applies a recurrent cell to the elements of the given input.

        Arguments
        ---------
        :param x: tensor
            Shape: (seq_len, batch_size, dim)
        :return:
            [1] output of shape (seq_len, batch, num_directions * hidden_size) containing the output features (h_t)
                from the last layer.
            [2] h_n of shape (num_layers * num_directions, batch, hidden_size) containing the hidden state for t=seq_len
            [3] c_n (if LSTM units are used) of shape (num_layers * num_directions, batch, hidden_size) contianing the
                cell state for t=seq_len
        """
        batch_size = x.shape[1]
        seq_length = x.shape[0]
        hidden = init_hidden(num_layers=self.num_layers, batch_size=batch_size, hidden_size=self.hidden_size,
                             num_dir=self.num_dir, dvc=x.device)
        if self.has_cell:
            cell = init_cell(num_layers=self.num_layers, batch_size=batch_size, hidden_size=self.hidden_size,
                             num_dir=self.num_dir, dvc=x.device)
        if self.has_stack:
            stack = init_stack(batch_size, self.stack_width, self.stack_depth, dvc=x.device)
        else:
            stack = None

        # Iteratively apply stack RNN to all characters in a sequence
        outputs = []
        if self.has_cell:
            hidden = (hidden, cell)
        for c in range(seq_length):
            if self.has_stack:
                stack_top = stack[:, 0, :]
                x_ = torch.cat([x[c, :, :], stack_top], dim=-1).unsqueeze(0)
                output, hidden = self.rnn(x_, hidden)

                # Stack update
                controls = torch.softmax(self.A_linear(output.view(batch_size, -1)), dim=-1)
                stack_input = torch.tanh(self.D_linear(output.view(batch_size, -1)))
                stack = self.stack_augmentation(stack_input, stack, controls)
            else:
                output, hidden = self.rnn(x, hidden)
            outputs.append(output)

        outputs = torch.cat(outputs)
        return outputs, hidden

    def stack_augmentation(self, input_val, prev_stack, controls):
        """
        Augmentation of the tensor into the stack. For more details see
        https://arxiv.org/abs/1503.01007

        Parameters
        ----------
        input_val: torch.tensor
            tensor to be added to stack

        prev_stack: torch.tensor
            previous stack state

        controls: torch.tensor
            predicted probabilities for each operation in the stack, i.e
            PUSH, POP and NO_OP. Again, see https://arxiv.org/abs/1503.01007

        Returns
        -------
        new_stack: torch.tensor
            new stack state
        """
        input_val = input_val.unsqueeze(1)
        batch_size = prev_stack.size(0)
        controls = controls.view(-1, 3, 1, 1)
        zeros_at_the_bottom = torch.zeros(batch_size, 1, self.stack_width)
        zeros_at_the_bottom = zeros_at_the_bottom.to(input_val.device)
        a_push, a_pop, a_no_op = controls[:, 0], controls[:, 1], controls[:, 2]
        stack_down = torch.cat((prev_stack[:, 1:], zeros_at_the_bottom), dim=1)
        stack_up = torch.cat((input_val, prev_stack[:, :-1]), dim=1)
        new_stack = a_no_op * prev_stack + a_push * stack_up + a_pop * stack_down
        return new_stack


class StackRNNLinear(nn.Module):
    """Linearly projects Stack RNN outputs to a fixed dimension"""

    def __init__(self, out_dim, hidden_size, bidirectional, encoder, bias=True, dropout=0.):
        super(StackRNNLinear, self).__init__()
        assert isinstance(encoder, Encoder)
        self.bias = bias
        if bidirectional:
            num_dir = 2
        else:
            num_dir = 1
        in_dim = hidden_size * num_dir
        self.linear_res = nn.Linear(in_dim, in_dim)
        if self.bias:
            self.bias_param = nn.Parameter(torch.zeros(out_dim))
        self.encoder = encoder
        self.sublayer = SublayerConnection(in_dim, dropout)

    def forward(self, rnn_input):
        """
        Takes the output of a Stack RNN and linearly projects each element.

        Arguments:
        ----------
        :param rnn_input: tuple
            A tuple where the RNN output is the first element of shape (seq_len, batch_size, num_dir * hidden_size)
        :return: tensor
            Shape: (seq_len, batch_size, out_dim)
        """
        x = rnn_input[0]
        x = self.sublayer(x, self.linear_res)
        weights = self.encoder.embeddings_weight()
        if self.bias:
            x = F.linear(x, weights, self.bias_param)
        else:
            x = F.linear(x, weights)
        return x
