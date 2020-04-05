import unittest
import numpy as np
import torch
from gpmt.data import GeneratorData
from gpmt.env import MoleculeEnv
from gpmt.model import Encoder, PositionalEncoding, StackDecoderLayer, LinearOut, StackRNN, StackRNNLinear
from gpmt.reward import RewardFunction
from gpmt.stackrnn import StackRNNCell
from gpmt.utils import init_hidden, init_stack, get_default_tokens, init_hidden_2d, init_stack_2d

gen_data_path = '../data/chembl_xsmall.smi'
# tokens = get_default_tokens()
# print(f'Number of tokens = {len(tokens)}')
gen_data = GeneratorData(training_data_path=gen_data_path, delimiter='\t',
                         cols_to_read=[0], keep_header=True, tokens=None, tokens_reload=True)

bz = 32


class MyTestCase(unittest.TestCase):

    def test_batch(self):
        batch = gen_data.random_training_set(batch_size=bz)
        assert (len(batch[0]) == bz and len(batch[1]) == bz)

    def test_embeddings(self):
        x, y = gen_data.random_training_set(batch_size=bz)
        encoder = Encoder(gen_data.n_characters, 128, gen_data.char2idx[gen_data.pad_symbol])
        x = encoder(x)
        assert (x.ndim == 3)
        print(f'x.shape = {x.shape}')

    def test_positional_encodings(self):
        x, y = gen_data.random_training_set(batch_size=bz)
        encoder = Encoder(gen_data.n_characters, 128, gen_data.char2idx[gen_data.pad_symbol])
        x = encoder(x)
        enc_shape = x.shape
        pe = PositionalEncoding(128, dropout=.2, max_len=500)
        x = pe(x)
        assert (x.shape == enc_shape)
        print(f'x.shape = {x.shape}')

    def test_stack_decoder_layer(self):
        x, y = gen_data.random_training_set(batch_size=bz)
        d_model = 128
        d_hidden = 10
        s_width = 16
        s_depth = 20
        encoder = Encoder(gen_data.n_characters, 128, gen_data.char2idx[gen_data.pad_symbol])
        x = encoder(x)
        pe = PositionalEncoding(d_model, dropout=.2, max_len=500)
        x = pe(x)
        h0 = init_hidden_2d(x.shape[1], x.shape[0], d_hidden)
        s0 = init_stack_2d(x.shape[1], x.shape[0], s_depth, s_width)
        stack_decoder = StackDecoderLayer(d_model=d_model, num_heads=1,
                                          d_hidden=d_hidden, stack_depth=s_depth,
                                          stack_width=s_width, dropout=.1)
        out = stack_decoder((x, s0))
        assert (len(out) == 3)

    def test_input_equals_output_embeddings(self):
        x, y = gen_data.random_training_set(batch_size=bz)
        encoder = Encoder(gen_data.n_characters, 128, gen_data.char2idx[gen_data.pad_symbol])
        lin_out = LinearOut(encoder.embeddings_weight)
        x = encoder(x)
        x_out = lin_out(x)
        assert x.shape == x_out.shape

    def test_stack_rnn_cell(self):
        x, y = gen_data.random_training_set(batch_size=bz)
        d_model = 128
        hidden_size = 16
        stack_width = 10
        stack_depth = 20
        num_layers = 1
        num_dir = 2
        encoder = Encoder(gen_data.n_characters, d_model, gen_data.char2idx[gen_data.pad_symbol])
        x = encoder(x)
        rnn_cells = []
        in_dim = d_model
        cell_type = 'gru'
        for _ in range(num_layers):
            rnn_cells.append(StackRNNCell(in_dim, hidden_size, has_stack=True,
                                          unit_type=cell_type, stack_depth=stack_depth,
                                          stack_width=stack_width))
            in_dim = hidden_size * num_dir
        rnn_cells = torch.nn.ModuleList(rnn_cells)

        h0 = init_hidden(num_layers=num_layers, batch_size=bz, hidden_size=hidden_size,
                         num_dir=num_dir)
        c0 = init_hidden(num_layers=num_layers, batch_size=bz, hidden_size=hidden_size, num_dir=num_dir)
        s0 = init_stack(bz, stack_width, stack_depth)

        seq_length = x.shape[0]
        hidden_outs = torch.zeros(num_layers, num_dir, seq_length, bz, hidden_size)
        if cell_type == 'lstm':
            cell_outs = torch.zeros(num_layers, num_dir, seq_length, bz, hidden_size)
        assert 0 <= num_dir <= 2
        for l in range(num_layers):
            for d in range(num_dir):
                h, c, stack = h0[l, d, :], c0[l, d, :], s0
                if d == 0:
                    indices = range(x.shape[0])
                else:
                    indices = reversed(range(x.shape[0]))
                for i in indices:
                    x_t = x[i, :, :]
                    hx, stack = rnn_cells[l](x_t, h, c, stack)
                    if cell_type == 'lstm':
                        hidden_outs[l, d, i, :, :] = hx[0]
                        cell_outs[l, d, i, :, :] = hx[1]
                    else:
                        hidden_outs[l, d, i, :, :] = hx

    def test_stack_rnn(self):
        x, y = gen_data.random_training_set(batch_size=bz)
        d_model = 128
        hidden_size = 16
        stack_width = 10
        stack_depth = 20
        num_layers = 2
        encoder = Encoder(gen_data.n_characters, d_model, gen_data.char2idx[gen_data.pad_symbol])
        x = encoder(x)
        stack_rnn = StackRNN(d_model, hidden_size, False, 'lstm', num_layers, stack_width, stack_depth,
                             dropout=0.2, k_mask_func=encoder.k_padding_mask)
        outputs = stack_rnn(x)
        assert len(outputs) > 1
        linear = StackRNNLinear(4, hidden_size, bidirectional=False)
        x = linear(outputs)
        print(x.shape)

    def test_mol_env(self):
        env = MoleculeEnv(gen_data, RewardFunction())
        print(f'sample action: {env.action_space.sample()}')
        print(f'sample observation: {env.observation_space.sample()}')
        s = env.reset()
        for i in range(5):
            env.render()
            action = env.action_space.sample()
            s_prime, reward, done, info = env.step(action)
            if done:
                env.reset()
                break

    def test_molecule_mcts(self):
        env = MoleculeEnv(gen_data, RewardFunction(reward_net=None,
                                                   policy=lambda x: gen_data.all_characters[
                                                       np.random.randint(gen_data.n_characters)],
                                                   actions=gen_data.all_characters))
        rewards = []
        for i in range(5):
            env.render()
            action = env.action_space.sample()
            s_prime, reward, done, info = env.step(action)
            rewards.append(reward)
            if done:
                env.reset()
                break
        print(f'rewards: {rewards}')


if __name__ == '__main__':
    unittest.main()
