import unittest
import numpy as np
import torch
from ptan.experience import ExperienceSourceFirstLast
from tqdm import tqdm

from gpmt.data import GeneratorData
from gpmt.env import MoleculeEnv
from gpmt.model import Encoder, PositionalEncoding, StackDecoderLayer, LinearOut, StackRNN, StackRNNLinear, RewardNetRNN
from gpmt.reward import RewardFunction
from gpmt.rl import PolicyAgent, MolEnvProbabilityActionSelector
from gpmt.stackrnn import StackRNNCell
from gpmt.utils import init_hidden, init_stack, get_default_tokens, init_hidden_2d, init_stack_2d, init_cell, seq2tensor

gen_data_path = '../data/chembl_xsmall.smi'
tokens = get_default_tokens()
# print(f'Number of tokens = {len(tokens)}')
gen_data = GeneratorData(training_data_path=gen_data_path, delimiter='\t',
                         cols_to_read=[0], keep_header=True, tokens=tokens, tokens_reload=True)

bz = 32


class MyTestCase(unittest.TestCase):

    def test_batch(self):
        batch = gen_data.random_training_set(batch_size=bz)
        assert (len(batch[0]) == bz and len(batch[1]) == bz)

    def test_embeddings(self):
        x, y = gen_data.random_training_set(batch_size=bz)
        encoder = Encoder(gen_data.n_characters, 128, gen_data.char2idx[gen_data.pad_symbol])
        x = encoder((x,))
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
        stack_decoder = StackDecoderLayer(d_model=d_model, num_heads=1, stack_depth=s_depth,
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
        d_model = 12
        hidden_size = 16
        stack_width = 10
        stack_depth = 20
        unit_type = 'lstm'
        num_layers = 2
        hidden_states = [get_initial_states(bz, hidden_size, 1, stack_depth, stack_width, unit_type)
                         for _ in range(num_layers)]
        encoder = Encoder(gen_data.n_characters, d_model, gen_data.char2idx[gen_data.pad_symbol])
        x = encoder(x)
        stack_rnn_1 = StackRNN(1, d_model, hidden_size, True, 'gru', stack_width, stack_depth,
                               k_mask_func=encoder.k_padding_mask)
        stack_rnn_2 = StackRNN(2, hidden_size, hidden_size, True, 'gru', stack_width, stack_depth,
                               k_mask_func=encoder.k_padding_mask)
        outputs = stack_rnn_1([x] + hidden_states)
        outputs = stack_rnn_2(outputs)
        assert len(outputs) > 1
        linear = StackRNNLinear(4, hidden_size, bidirectional=False, )
        x = linear(outputs)
        print(x[0].shape)

    def test_mol_env(self):
        d_model = 8
        hidden_size = 16
        num_layers = 1
        encoder = Encoder(gen_data.n_characters, d_model, gen_data.char2idx[gen_data.pad_symbol], return_tuple=True)
        rnn = RewardNetRNN(d_model, hidden_size, num_layers, bidirectional=True, unit_type='gru')
        reward_net = torch.nn.Sequential(encoder, rnn)
        env = MoleculeEnv(gen_data, RewardFunction(reward_net=reward_net,
                                                   policy=lambda x: gen_data.all_characters[
                                                       np.random.randint(gen_data.n_characters)],
                                                   actions=gen_data.all_characters))
        print(f'sample action: {env.action_space.sample()}')
        print(f'sample observation: {env.observation_space.sample()}')
        s = env.reset()
        for i in range(5):
            env.render()
            action = env.action_space.sample()
            print(f'action = {action}')
            s_prime, reward, done, info = env.step(action)
            if done:
                env.reset()
                break

    def test_molecule_mcts(self):
        d_model = 8
        hidden_size = 16
        num_layers = 2
        encoder = Encoder(gen_data.n_characters, d_model, gen_data.char2idx[gen_data.pad_symbol],
                          return_tuple=False)
        rnn = RewardNetRNN(d_model, hidden_size, num_layers, bidirectional=True, unit_type='gru')
        env = MoleculeEnv(gen_data, RewardFunction(reward_net=torch.nn.Sequential(encoder, rnn),
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

    def test_reward_rnn(self):
        x, y = gen_data.random_training_set(batch_size=bz)
        d_model = 8
        hidden_size = 16
        num_layers = 2
        encoder = Encoder(gen_data.n_characters, d_model, gen_data.char2idx[gen_data.pad_symbol],
                          return_tuple=False)
        x = encoder([x])
        rnn = RewardNetRNN(d_model, hidden_size, num_layers, bidirectional=True, unit_type='lstm')
        r = rnn(x)
        print(f'reward: {r}')

    def test_policy_net(self):
        d_model = 8
        hidden_size = 16
        num_layers = 1
        stack_width = 10
        stack_depth = 20
        unit_type = 'lstm'

        # Create a function to provide initial hidden states
        def hidden_states_func():
            return [get_initial_states(1, hidden_size, 1, stack_depth, stack_width, unit_type) for _ in
                    range(num_layers)]

        # Encoder to map character indices to embeddings
        encoder = Encoder(gen_data.n_characters, d_model, gen_data.char2idx[gen_data.pad_symbol], return_tuple=True)

        # Reward function model
        rnn = RewardNetRNN(d_model, hidden_size, num_layers, bidirectional=True, unit_type='gru')
        reward_net = torch.nn.Sequential(encoder, rnn)
        reward_function = RewardFunction(reward_net=reward_net, policy=lambda x: gen_data.all_characters[
            np.random.randint(gen_data.n_characters)], actions=gen_data.all_characters)

        # Create molecule generation environment
        env = MoleculeEnv(gen_data, reward_function)

        # Create agent network
        stack_rnn = StackRNN(1, d_model, hidden_size, True, 'lstm', stack_width, stack_depth,
                             k_mask_func=encoder.k_padding_mask)
        stack_linear = StackRNNLinear(gen_data.n_characters, hidden_size, bidirectional=False)
        agent_net = torch.nn.Sequential(encoder, stack_rnn, stack_linear)

        # Create agent
        selector = MolEnvProbabilityActionSelector(actions=gen_data.all_characters)
        agent = PolicyAgent(model=agent_net,
                            action_selector=selector,
                            states_preprocessor=seq2tensor,
                            initial_state=hidden_states_func,
                            apply_softmax=True,
                            device='cpu')

        # Ptan ops for aggregating experiences
        exp_source = ExperienceSourceFirstLast(env, agent, gamma=0.97)

        # Begin simulation and training
        batch_states, batch_actions, batch_qvals = [], [], []
        for step_idx, exp in enumerate(exp_source):
            batch_states.append(exp.state)
            batch_actions.append(exp.action)
            print(f'state = {exp.state}, action = {exp.action}, reward = {exp.reward}, next_state = {exp.last_state}')
            if step_idx == 5:
                break


def get_initial_states(batch_size, hidden_size, num_layers, stack_depth, stack_width, unit_type):
    hidden = init_hidden(num_layers=num_layers, batch_size=batch_size, hidden_size=hidden_size, num_dir=1, dvc='cpu')
    if unit_type == 'lstm':
        cell = init_cell(num_layers=num_layers, batch_size=batch_size, hidden_size=hidden_size, num_dir=1, dvc='cpu')
    else:
        cell = None
    stack = init_stack(batch_size, stack_width, stack_depth, dvc='cpu')
    return hidden, cell, stack


if __name__ == '__main__':
    unittest.main()
