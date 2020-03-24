# Author: bbrighttaer
# Project: GPMT
# Date: 3/22/2020
# Time: 9:13 AM
# File: model.py

from __future__ import absolute_import, division, print_function, unicode_literals

import math
import torch
import torch.nn as nn

from gpmt.utils import init_hidden, init_stack


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, padding_idx):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

    def forward(self, x):
        """
        Retrieve embeddings for x using the indices.
        :param x: tensor
            x.shape = (batch_size, sequence length)
        :return: tensor
            x of shape (sequence length, batch_size, d_model)
        """
        x = self.embedding(x)
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
        self.nonlinearity = NonsatActivation()
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
        x = self.dropout(self.nonlinearity(self.normalize(self.linear(x))))
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


class StackDecoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, d_hidden, stack_depth, stack_width, d_ff=2048, d_ss=128, dropout=0.):
        super(StackDecoderLayer, self).__init__()
        self.d_hidden = d_hidden
        self.stack_depth = stack_depth
        self.stack_width = stack_width
        self.multihead_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)
        self.mha_normalize = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn_inner = nn.Linear(d_model, d_ff)
        self.ffn_outer = nn.Linear(d_ff, d_model)
        self.nonlinearity = NonsatActivation()

        # stack & hidden state elements
        self.W = nn.Linear(d_model, d_hidden)
        self.R = nn.Linear(d_hidden, d_hidden)
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
        x_in, hidden_prev, stack_prev = inp
        seq_length, batch_size, _ = x_in.shape
        mask = _create_attn_mask(seq_length, x_in.device)

        # masked multihead self-attention
        x, _ = self.multihead_attention(x_in, x_in, x_in, attn_mask=mask, need_weights=False)
        x = x_in + x
        x = self.mha_normalize(x)
        x = self.nonlinearity(x)

        # hidden state ops
        stack_x = stack_prev[:, :, 0, :].view(batch_size, seq_length, -1)
        hidden = self.W(x.permute(1, 0, 2)) + self.R(hidden_prev) + self.P(stack_x)
        hidden = self.nonlinearity(hidden)

        # stack ops
        vx = torch.cat([x, hidden.permute(1, 0, 2)], dim=-1)
        y = self.U(torch.softmax(self.V(vx), dim=-1))

        # pooling
        x = torch.stack([x, y])
        x = torch.mean(x, dim=0)

        # stack update
        stack_inp = self.nonlinearity(self.D(hidden)).unsqueeze(2)
        stack_controls = torch.softmax(self.A(hidden), dim=-1)
        stack = self.stack_augmentation(stack_inp, stack_prev, stack_controls)

        # FFN module
        x = self.ffn_norm(x + self.ffn_outer(self.nonlinearity(self.ffn_inner(x))))
        return x, hidden, stack

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

    def __init__(self, d_hidden, s_depth, s_width):
        super(AttentionInitialize, self).__init__()
        self.d_hidden = d_hidden
        self.s_depth = s_depth
        self.s_width = s_width

    def forward(self, x):
        """

        :param x: tensor
            Encoded input of shape (Seq. length, batch_size, d_model)
        :return:
        """
        h0 = init_hidden(x.shape[1], x.shape[0], self.d_hidden)
        s0 = init_stack(x.shape[1], x.shape[0], self.s_depth, self.s_width)
        return x, h0, s0


class AttentionTerminal(nn.Module):
    def forward(self, inp):
        """Prepares the final attention output before applying feeding to classifier."""
        return inp[0]
