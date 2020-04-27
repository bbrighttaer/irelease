# Author: bbrighttaer
# Project: GPMT
# Date: 3/22/2020
# Time: 9:13 AM
# File: model.py

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpmt.utils import init_hidden, init_cell


def clone(module, N):
    """
    Make N copies of an nn.Module

    :param module: the model to copy
    :param N: the number of copies
    :return: an nn.ModuleList of the copies
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, padding_idx, dropout=0., return_tuple=False):
        """
        Provides the embeddings of characters.

        Arguments
        ---------
        :param vocab_size: int
            The size of the vocabulary.
        :param d_model: int
            The dimension of an embedding.
        :param padding_idx: int
            The index in the vocabulary that would be used for padding.
        :param dropout: float
            The dropout probability.
        """
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.padding_idx = padding_idx
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.k_mask = None
        self.return_tup = return_tuple

    def k_padding_mask(self):
        return self.k_mask

    def embeddings_weight(self):
        return self.embedding.weight

    def forward(self, inp):
        """
        Retrieve embeddings for x using the indices.
        :param inp: tuple or tensor (x)
            Index 0 contains x: x.shape = (batch_size, sequence length)
        :return: tuple or tensor
            Updates x to have the shape (sequence length, batch_size, d_model)
        """
        is_list = False
        if isinstance(inp, list):
            x = inp[0]
            is_list = True
        else:
            x = inp
        self.k_mask = x == self.padding_idx
        # x = self.dropout(self.embedding(x) * math.sqrt(self.d_model))
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        if self.return_tup:
            if not is_list:
                inp = [None]
            inp[0] = x
            return inp
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

    def forward(self, inp):
        """
        Concatenates the positional encodings to the input.
        Assumes x is organized as: (Length, batch size, d_model)
        :param inp: tuple
            First element in the tuple is x
        :return: tuple
            Updates x/index 0 of inp to have the shape (Length, batch size, d_model)
        """
        x = inp[0]
        seq_length, batch_size, _ = x.shape
        x_pe = self.pe[:seq_length, :]
        x_pe = torch.repeat_interleave(x_pe.unsqueeze(1), batch_size, dim=1)
        x = torch.cat([x, x_pe], dim=-1)
        x = self.dropout(torch.relu(self.normalize(self.linear(x))))
        inp[0] = x
        return inp


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


class SublayerConnectionQKV(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnectionQKV, self).__init__()
        self.norm_kv = nn.LayerNorm(d_model)
        self.norm_q = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, kv, sublayer):
        q = self.norm_q(q)
        k = v = self.norm_kv(kv)
        x = v + self.dropout(sublayer((q, k, v)))
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

    def __init__(self, d_model, num_heads, stack_depth, stack_width, k_mask_func, d_ff=2048, dropout=0.,
                 use_memory=True):
        super(StackDecoderLayer, self).__init__()
        assert callable(k_mask_func), 'k_mask_func argument should be a method of the ' \
                                      'Encoder that provides the key padding mask.'
        self.use_memory = use_memory
        self.d_hidden = d_model
        self.stack_depth = stack_depth
        self.stack_width = stack_width
        self.k_padding_mask_func = k_mask_func
        self.multihead_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.feedforwad = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.subalyers = nn.ModuleList([SublayerConnection(d_model, dropout), SublayerConnection(d_model, dropout)])

        # stack & hidden state elements
        self.nonlinearity = nn.ReLU()  # NonsatActivation()
        self.W = nn.Linear(d_model, self.d_hidden)
        self.P = nn.Linear(stack_width, self.d_hidden)
        self.A = nn.Linear(self.d_hidden, 3)
        self.D = nn.Linear(self.d_hidden, stack_width)

    def forward(self, inp):
        """
        Performs a forward pass through the stack-augmented Transformer-Decoder.

        :param inp: tuple
            [0]: the input (seq. length, batch_size, d_model)
            [1]: the previous stack. (batch_size, seq. length, stack_depth, stack_width).
        :return: tuple of tensors
            [0]: the transformed input (seq. length, batch_size, d_model)
            [1]: the updated stack. (batch_size, seq. length, stack_depth, stack_width).
        """
        x_in, stack_prev = inp
        seq_length, batch_size, _ = x_in.shape
        mask = _create_attn_mask(seq_length, x_in.device)
        k_mask = self.k_padding_mask_func()

        stack = stack_prev
        if self.use_memory:
            # hidden state ops
            stack_x = stack_prev[:, :, 0, :].view(batch_size, seq_length, -1)
            hidden = self.W(x_in.permute(1, 0, 2)) + self.P(stack_x)
            hidden = self.nonlinearity(hidden)

            # stack update
            stack_inp = torch.tanh(self.D(hidden)).unsqueeze(2)
            stack_controls = torch.softmax(self.A(hidden), dim=-1)
            stack = self.stack_augmentation(stack_inp, stack_prev, stack_controls)

            x_in = x_in + hidden.permute(1, 0, 2)

        # masked multihead self-attention
        x = self.subalyers[0](x_in, lambda w: self.multihead_attention(w, w, w, key_padding_mask=k_mask,
                                                                       attn_mask=mask, need_weights=False)[0])
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


# class AttentionInitialize(nn.Module):
#     """Prepares the encoded input for propagation through the memory layer(s)."""
#
#     def __init__(self, d_hidden, s_depth, s_width, dvc='cpu'):
#         super(AttentionInitialize, self).__init__()
#         self.d_hidden = d_hidden
#         self.s_depth = s_depth
#         self.s_width = s_width
#         self.dvc = dvc
#
#     def forward(self, x):
#         """
#
#         :param x: tensor
#             Encoded input of shape (Seq. length, batch_size, d_model)
#         :return:
#         """
#         # h0 = init_hidden_2d(x.shape[1], x.shape[0], self.d_hidden, dvc=self.dvc)
#         s0 = init_stack_2d(x.shape[1], x.shape[0], self.s_depth, self.s_width, dvc=self.dvc)
#         return x, s0


class AttentionTerminal(nn.Module):
    def __init__(self, need_stack=False):
        super(AttentionTerminal, self).__init__()
        self.need_stack = need_stack

    def forward(self, inp):
        """
        Prepares the final attention output before applying feeding to classifier.

        Arguments:
        ---------
        :param inp: tuple of (x, stack)
        :return:
        """
        if self.need_stack:
            return inp
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


class StackRNN(nn.Module):
    def __init__(self, layer_index, input_size, hidden_size, has_stack, unit_type='lstm', stack_width=None,
                 stack_depth=None, bias=True, k_mask_func=None):
        super(StackRNN, self).__init__()
        self.layer_index = layer_index
        self.hidden_size = int(hidden_size)
        self.has_stack = has_stack
        self.has_cell = True if unit_type == 'lstm' else False
        self.stack_width = int(stack_width)
        self.stack_depth = int(stack_depth)
        self.unit_type = unit_type
        self.num_dir = 1
        self.k_padding_mask_func = k_mask_func
        if has_stack:
            input_size = int(input_size + stack_width)
            self.A_linear = nn.Linear(hidden_size, 3)
            self.D_linear = nn.Linear(hidden_size, stack_width)
        else:
            input_size = int(input_size)
        if self.unit_type == 'lstm':
            self.rnn = nn.LSTM(input_size, self.hidden_size, 1, bidirectional=False, bias=bias)
        elif self.unit_type == 'gru':
            self.rnn = nn.GRU(input_size, self.hidden_size, 1, bidirectional=False, bias=bias)

    def forward(self, inp, **kwargs):
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
        x, hidden_states = inp[0], inp[self.layer_index]
        hidden, cell, stack = hidden_states
        batch_size = x.shape[1]
        seq_length = x.shape[0]
        hidden_shape = hidden.shape

        if self.has_cell:
            hidden = (hidden, cell)
        # Iteratively apply stack RNN to all characters in a sequence
        outputs = []
        for c in range(seq_length):
            if self.has_stack:
                # Stack update
                if self.has_cell:
                    hidden_2_stack = hidden[0].view(*hidden_shape)
                else:
                    hidden_2_stack = hidden
                hidden_2_stack = hidden_2_stack.permute(1, 0, 2).contiguous().view(batch_size, -1)
                controls = torch.softmax(self.A_linear(hidden_2_stack), dim=-1)
                stack_input = torch.tanh(self.D_linear(hidden_2_stack))
                stack = self.stack_augmentation(stack_input, stack, controls)

                # rnn
                stack_top = stack[:, 0, :].unsqueeze(0)
                x_ = x[c, :, :].unsqueeze(0)
                x_ = torch.cat([x_, stack_top], dim=-1)
                output, hidden = self.rnn(x_, hidden)
            else:
                output, hidden = self.rnn(x, hidden)
            outputs.append(output)
        x = torch.cat(outputs)
        inp[0] = x
        if self.has_cell:
            inp[self.layer_index] = (hidden[0], hidden[1], stack)
        else:
            inp[self.layer_index] = (hidden, cell, stack)
        return inp

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
        stack_down = torch.cat((prev_stack[:, 1:, :], zeros_at_the_bottom), dim=1)
        stack_up = torch.cat((input_val, prev_stack[:, :-1, :]), dim=1)
        new_stack = a_no_op * prev_stack + a_push * stack_up + a_pop * stack_down
        return new_stack


class StackRNNLinear(nn.Module):
    """Linearly projects Stack RNN outputs to a fixed dimension"""

    def __init__(self, out_dim, hidden_size, bidirectional, bias=True):  # , encoder, bias=True, dropout=0.):
        super(StackRNNLinear, self).__init__()
        # assert isinstance(encoder, Encoder)
        self.bias = bias
        if bidirectional:
            num_dir = 2
        else:
            num_dir = 1
        in_dim = hidden_size * num_dir
        self.decoder = nn.Linear(in_dim, out_dim)
        # self.linear_res = nn.Linear(in_dim, in_dim)
        # if self.bias:
        #     self.bias_param = nn.Parameter(torch.zeros(out_dim))
        # self.encoder = encoder
        # self.sublayer = SublayerConnection(in_dim, dropout)

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
        # x = self.sublayer(x, self.linear_res)
        # weights = self.encoder.embeddings_weight()
        # if self.bias:
        #     x = F.linear(x, weights, self.bias_param)
        # else:
        #     x = F.linear(x, weights)
        rnn_input[0] = self.decoder(x)
        return rnn_input


class StackedRNNDropout(nn.Module):
    def __init__(self, p):
        super(StackedRNNDropout, self).__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, inp):
        inp[0] = self.dropout(inp[0])
        return inp


class StackedRNNLayerNorm(nn.Module):
    def __init__(self, dim):
        super(StackedRNNLayerNorm, self).__init__()
        self.normalize = nn.LayerNorm(dim)

    def forward(self, inp):
        inp[0] = self.normalize(inp[0])
        return inp


class RewardNetRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=True, dropout=0., unit_type='lstm'):
        super(RewardNetRNN, self).__init__()
        self.num_dir = 1
        self.num_layers = num_layers
        if bidirectional:
            self.num_dir += 1
        self.hidden_size = hidden_size
        if unit_type == 'lstm':
            self.has_cell = True
            self.base_rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)
            self.post_rnn = nn.LSTMCell((hidden_size * self.num_dir) + hidden_size, hidden_size)
        else:
            self.has_cell = False
            self.base_rnn = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)
            self.post_rnn = nn.GRUCell((hidden_size * self.num_dir) + hidden_size, hidden_size)
        self.linear = nn.Linear((hidden_size * self.num_dir) + hidden_size, 1)
        self.reward_net = nn.Sequential(nn.LayerNorm(hidden_size + 1),
                                        nn.Linear(self.hidden_size + 1, 1))

    def forward(self, inp):
        """
        Evaluates the input to determine its reward.

        Argument
        :param inp: list / tuple
           [0] Input from encoder of shape (seq_len, batch_size, embed_dim)
           [1] SMILES validity flag
        :return: tensor
            Reward of shape (batch_size, 1)
        """
        x = inp[0]
        seq_len, batch_size = x.shape[:2]

        # Construct initial states
        hidden = init_hidden(self.num_layers, batch_size, self.hidden_size, self.num_dir, x.device)
        hidden_ = init_hidden(1, batch_size, self.hidden_size, 1, x.device).view(batch_size, self.hidden_size)
        if self.has_cell:
            cell = init_cell(self.num_layers, batch_size, self.hidden_size, self.num_dir, x.device)
            cell_ = init_cell(1, batch_size, self.hidden_size, 1, x.device).view(batch_size, self.hidden_size)
            hidden, hidden_ = (hidden, cell), (hidden_, cell_)

        # Apply base rnn
        output, hidden = self.base_rnn(x, hidden)

        # Additive attention, see: http://arxiv.org/abs/1409.0473
        for i in range(seq_len):
            h = hidden_[0] if self.has_cell else hidden_
            s = h.unsqueeze(0).expand(seq_len, *h.shape)
            x_ = torch.cat([output, s], dim=-1)
            logits = self.linear(x_.contiguous().view(-1, x_.shape[-1]))
            wts = torch.softmax(logits.view(seq_len, batch_size).t(), -1).unsqueeze(2)
            x_ = x_.permute(1, 2, 0)
            ctx = x_.bmm(wts).squeeze(dim=2)
            hidden_ = self.post_rnn(ctx, hidden_)
        rw_x = hidden_[0] if self.has_cell else hidden_
        rw_x = torch.cat([rw_x, inp[-1]], dim=-1)
        reward = self.reward_net(rw_x)
        return reward


class CriticRNN(nn.Module):
    def __init__(self, input_size, hidden_size, unit_type='gru', num_layers=1):
        super(CriticRNN, self).__init__()
        rnn = nn.GRU if unit_type == 'gru' else nn.LSTM
        self.has_cell = unit_type == 'lstm'
        self.hidden_size = hidden_size
        self.rnn = rnn(input_size, hidden_size, num_layers, bidirectional=True)
        self.num_layers = 1
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.linear = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        """
        Critic net
        :param x: tensor
            x.shape structure is (seq. len, batch, dim)
        :return: tensor
            (seq_len/states, batch, 1)
        """
        if isinstance(x, (list, tuple)):
            x = x[0]
        batch_size = x.shape[1]
        hidden = init_hidden(self.num_layers, batch_size, self.hidden_size, 2, x.device)
        if self.has_cell:
            cell = init_cell(self.num_layers, x.shape[1], self.hidden_size, 2, x.device)
            hidden = (hidden, cell)
        x, _ = self.rnn(x, hidden)
        x = self.linear(self.norm(x))
        return x
