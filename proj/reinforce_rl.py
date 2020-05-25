# Author: bbrighttaer
# Project: GPMT
# Date: 4/8/2020
# Time: 8:02 PM
# File: reinforce_rl.py

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import copy
import os
import random
import time
from datetime import datetime as dt

import numpy as np
import torch
import torch.nn as nn
from ptan.common.utils import TBMeanTracker
from ptan.experience import ExperienceSourceFirstLast
from soek import Trainer, DataNode
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from gpmt.data import GeneratorData
from gpmt.env import MoleculeEnv
from gpmt.model import Encoder, StackRNN, StackRNNLinear, RewardNetRNN
from gpmt.predictor import RNNPredictor
from gpmt.reward import RewardFunction
from gpmt.rl import MolEnvProbabilityActionSelector, PolicyAgent, GuidedRewardLearningIRL, \
    StateActionProbRegistry, REINFORCE, Trajectory, EpisodeStep
from gpmt.utils import Flags, get_default_tokens, parse_optimizer, seq2tensor, init_hidden, init_cell, init_stack, \
    time_since, generate_smiles

currentDT = dt.now()
date_label = currentDT.strftime("%Y_%m_%d__%H_%M_%S")

seeds = [1]

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


class IReLeaSE(Trainer):

    @staticmethod
    def initialize(hparams, demo_data_gen, unbiased_data_gen, prior_data_gen, *args, **kwargs):
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
            # rnn_layers.append(StackedRNNDropout(hparams['dropout']))
            # rnn_layers.append(StackedRNNLayerNorm(hparams['d_model']))
        agent_net = nn.Sequential(encoder,
                                  *rnn_layers,
                                  StackRNNLinear(out_dim=demo_data_gen.n_characters,
                                                 hidden_size=hparams['d_model'],
                                                 bidirectional=False,
                                                 bias=True))
        agent_net = agent_net.to(device)
        optimizer_agent_net = parse_optimizer(hparams['agent_params'], agent_net)
        selector = MolEnvProbabilityActionSelector(actions=demo_data_gen.all_characters)
        probs_reg = StateActionProbRegistry()
        init_state_args = {'num_layers': hparams['agent_params']['num_layers'],
                           'hidden_size': hparams['d_model'],
                           'stack_depth': hparams['agent_params']['stack_depth'],
                           'stack_width': hparams['agent_params']['stack_width'],
                           'unit_type': hparams['agent_params']['unit_type']}
        agent = PolicyAgent(model=agent_net,
                            action_selector=selector,
                            states_preprocessor=seq2tensor,
                            initial_state=agent_net_hidden_states_func,
                            initial_state_args=init_state_args,
                            apply_softmax=True,
                            probs_registry=probs_reg,
                            device=device)
        drl_alg = REINFORCE(model=agent_net, optimizer=optimizer_agent_net,
                            initial_states_func=agent_net_hidden_states_func,
                            initial_states_args=init_state_args,
                            prior_data_gen=prior_data_gen,
                            device=device,
                            xent_lambda=hparams['xent_lambda'],
                            gamma=hparams['gamma'],
                            grad_clipping=hparams['reinforce_max_norm'],
                            lr_decay_gamma=hparams['lr_decay_gamma'],
                            lr_decay_step=hparams['lr_decay_step_size'])

        # Reward function entities
        reward_net = nn.Sequential(encoder,
                                   RewardNetRNN(input_size=hparams['d_model'],
                                                hidden_size=hparams['reward_params']['d_model'],
                                                num_layers=hparams['reward_params']['num_layers'],
                                                bidirectional=True,
                                                dropout=hparams['dropout'],
                                                unit_type=hparams['reward_params']['unit_type']))
        reward_net = reward_net.to(device)

        expert_model = RNNPredictor(hparams['expert_model_params'], device)
        reward_function = RewardFunction(reward_net, mc_policy=agent, actions=demo_data_gen.all_characters,
                                         device=device, use_mc=hparams['use_monte_carlo_sim'],
                                         mc_max_sims=hparams['monte_carlo_N'],
                                         expert_func=expert_model,
                                         no_mc_fill_val=hparams['no_mc_fill_val'])
        optimizer_reward_net = parse_optimizer(hparams['reward_params'], reward_net)
        demo_data_gen.set_batch_size(hparams['reward_params']['demo_batch_size'])
        irl_alg = GuidedRewardLearningIRL(reward_net, optimizer_reward_net, demo_data_gen,
                                          k=hparams['reward_params']['irl_alg_num_iter'],
                                          agent_net=agent_net,
                                          agent_net_init_func=agent_net_hidden_states_func,
                                          agent_net_init_func_args=init_state_args,
                                          device=device)

        init_args = {'agent': agent,
                     'probs_reg': probs_reg,
                     'drl_alg': drl_alg,
                     'irl_alg': irl_alg,
                     'reward_func': reward_function,
                     'gamma': hparams['gamma'],
                     'episodes_to_train': hparams['episodes_to_train'],
                     'expert_model': expert_model,
                     'demo_data_gen': demo_data_gen,
                     'unbiased_data_gen': unbiased_data_gen,
                     'gen_args': {'num_layers': hparams['agent_params']['num_layers'],
                                  'hidden_size': hparams['d_model'],
                                  'num_dir': 1,
                                  'stack_depth': hparams['agent_params']['stack_depth'],
                                  'stack_width': hparams['agent_params']['stack_width'],
                                  'has_stack': has_stack,
                                  'has_cell': hparams['agent_params']['unit_type'] == 'lstm',
                                  'device': device}}
        return init_args

    @staticmethod
    def data_provider(k, flags):
        tokens = get_default_tokens()
        demo_data = GeneratorData(training_data_path=flags.demo_file,
                                  delimiter='\t',
                                  cols_to_read=[0],
                                  keep_header=True,
                                  pad_symbol=' ',
                                  max_len=120,
                                  tokens=tokens,
                                  use_cuda=use_cuda)
        unbiased_data = GeneratorData(training_data_path=flags.unbiased_file,
                                      delimiter='\t',
                                      cols_to_read=[0],
                                      keep_header=True,
                                      pad_symbol=' ',
                                      max_len=120,
                                      tokens=tokens,
                                      use_cuda=use_cuda)
        prior_data = GeneratorData(training_data_path=flags.prior_data,
                                   delimiter='\t',
                                   cols_to_read=[0],
                                   keep_header=True,
                                   pad_symbol=' ',
                                   max_len=120,
                                   tokens=tokens,
                                   use_cuda=use_cuda)
        return {'demo_data': demo_data, 'unbiased_data': unbiased_data, 'prior_data': prior_data}

    @staticmethod
    def evaluate(*args, **kwargs):
        super().evaluate(*args, **kwargs)

    @staticmethod
    def train(init_args, agent_net_path=None, agent_net_name=None, seed=0, n_episodes=5000, sim_data_node=None,
              tb_writer=None, is_hsearch=False):
        tb_writer = tb_writer()
        agent = init_args['agent']
        probs_reg = init_args['probs_reg']
        drl_algorithm = init_args['drl_alg']
        irl_algorithm = init_args['irl_alg']
        reward_func = init_args['reward_func']
        gamma = init_args['gamma']
        episodes_to_train = init_args['episodes_to_train']
        expert_model = init_args['expert_model']
        demo_data_gen = init_args['demo_data_gen']
        unbiased_data_gen = init_args['unbiased_data_gen']
        best_model_wts = None
        best_score = 0.

        # load pretrained model
        if agent_net_path and agent_net_name:
            print('Loading pretrained model...')
            agent.model.load_state_dict(IReLeaSE.load_model(agent_net_path, agent_net_name))
            print('Pretrained model loaded successfully!')

        start = time.time()

        # Begin simulation and training
        total_rewards = []
        irl_trajectories = []
        done_episodes = 0
        batch_episodes = 0
        exp_trajectories = []

        env = MoleculeEnv(actions=get_default_tokens(), reward_func=reward_func)
        exp_source = ExperienceSourceFirstLast(env, agent, gamma, steps_count=1, steps_delta=1)
        traj_prob = 1.
        exp_traj = []

        demo_score = np.mean(expert_model(demo_data_gen.random_training_set_smiles(1000))[1])
        baseline_score = np.mean(expert_model(unbiased_data_gen.random_training_set_smiles(1000))[1])
        n_to_generate = 200
        with TBMeanTracker(tb_writer, 1) as tracker:
            for step_idx, exp in tqdm(enumerate(exp_source)):
                exp_traj.append(exp)
                traj_prob *= probs_reg.get(list(exp.state), exp.action)

                if exp.last_state is None:
                    irl_trajectories.append(Trajectory(terminal_state=EpisodeStep(exp.state, exp.action),
                                                       traj_prob=traj_prob))
                    exp_trajectories.append(exp_traj)  # for ExperienceFirstLast objects
                    exp_traj = []
                    traj_prob = 1.
                    probs_reg.clear()
                    batch_episodes += 1

                new_rewards = exp_source.pop_total_rewards()
                if new_rewards:
                    reward = new_rewards[0]
                    done_episodes += 1
                    total_rewards.append(reward)
                    mean_rewards = float(np.mean(total_rewards[-100:]))
                    tracker.track('mean_total_reward', mean_rewards, step_idx)
                    tracker.track('total_reward', reward, step_idx)
                    print(f'Time = {time_since(start)}, step = {step_idx}, reward = {reward:6.2f}, '
                          f'mean_100 = {mean_rewards:6.2f}, episodes = {done_episodes}')
                    with torch.set_grad_enabled(False):
                        samples = generate_smiles(drl_algorithm.model, demo_data_gen, init_args['gen_args'],
                                                  num_samples=n_to_generate)
                    predictions = expert_model(samples)[1]
                    score = np.mean(predictions)
                    percentage_in_threshold = np.sum((predictions >= 0.0) & (predictions <= 5.0)) / len(predictions)
                    per_valid = len(predictions) / n_to_generate
                    print(f'Mean value of predictions = {score}, '
                          f'% of valid SMILES = {per_valid}, '
                          f'% in drug-like region={percentage_in_threshold}')
                    tb_writer.add_scalars('qsar_score', {'sampled': score,
                                                         'baseline': baseline_score,
                                                         'demo_data': demo_score}, step_idx)
                    tb_writer.add_scalars('SMILES stats', {'per. of valid': per_valid,
                                                           'per. in drug-like region': percentage_in_threshold},
                                          step_idx)
                    if score >= best_score:
                        best_model_wts = [copy.deepcopy(drl_algorithm.model.state_dict()),
                                          copy.deepcopy(irl_algorithm.model.state_dict())]
                        best_score = score

                    if done_episodes == n_episodes:
                        print('Training completed!')
                        break

                if batch_episodes < episodes_to_train:
                    continue

                # Train models
                print('Fitting models...')
                irl_loss = irl_algorithm.fit(irl_trajectories)
                rl_loss = drl_algorithm.fit(exp_trajectories)
                samples = generate_smiles(drl_algorithm.model, demo_data_gen, init_args['gen_args'],
                                          num_samples=3)
                print(f'IRL loss = {irl_loss}, RL loss = {rl_loss}, samples = {samples}')
                tracker.track('irl_loss', irl_loss, step_idx)
                tracker.track('agent_loss', rl_loss, step_idx)

                # Reset
                batch_episodes = 0
                irl_trajectories.clear()
                exp_trajectories.clear()

        drl_algorithm.model.load_state_dict(best_model_wts[0])
        irl_algorithm.model.load_state_dict(best_model_wts[1])
        duration = time.time() - start
        print('\nTraining duration: {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))
        return {'model': [drl_algorithm.model, irl_algorithm.model],
                'score': round(best_score, 3),
                'epoch': done_episodes}

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
        return torch.load(os.path.join(path, name), map_location=torch.device(device))


def main(flags):
    sim_label = 'DeNovo-IReLeaSE-reinf'
    sim_data = DataNode(label=sim_label)
    nodes_list = []
    sim_data.data = nodes_list

    # For searching over multiple seeds
    hparam_search = None

    for seed in seeds:
        summary_writer_creator = lambda: SummaryWriter(log_dir="irelease"
                                                               "/{}_{}_{}/".format(sim_label, seed, dt.now().strftime(
            "%Y_%m_%d__%H_%M_%S")))

        # for data collection of this round of simulation.
        data_node = DataNode(label="seed_%d" % seed)
        nodes_list.append(data_node)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        print('--------------------------------------------------------------------------------')
        print(f'{device}\n{sim_label}\tDemonstrations file: {flags.demo_file}')
        print('--------------------------------------------------------------------------------')

        irelease = IReLeaSE()
        k = 1
        if flags.hparam_search:
            pass
        else:
            hyper_params = default_hparams(flags)
            data_gens = irelease.data_provider(k, flags)
            init_args = irelease.initialize(hyper_params, data_gens['demo_data'], data_gens['unbiased_data'],
                                            data_gens['prior_data'])
            results = irelease.train(init_args, flags.model_dir, flags.pretrained_model, seed,
                                     sim_data_node=data_node,
                                     n_episodes=10000,
                                     tb_writer=summary_writer_creator)
            irelease.save_model(results['model'][0],
                                path=flags.model_dir,
                                name=f'irelease_stack-rnn_{hyper_params["agent_params"]["unit_type"]}_reinforce_agent_'
                                     f'{date_label}_{results["score"]}_{results["epoch"]}')
            irelease.save_model(results['model'][1],
                                path=flags.model_dir,
                                name=f'irelease_stack-rnn_{hyper_params["agent_params"]["unit_type"]}_reward_net_'
                                     f'{date_label}_{results["score"]}_{results["epoch"]}')

    # save simulation data resource tree to file.
    sim_data.to_json(path="./analysis/")


def default_hparams(args):
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
            'reward_params': {'num_layers': 2,
                              'd_model': 256,
                              'unit_type': 'lstm',
                              'demo_batch_size': 32,
                              'irl_alg_num_iter': 5,
                              'optimizer': 'adadelta',
                              'optimizer__global__weight_decay': 0.0000,
                              'optimizer__global__lr': 0.001, },
            'agent_params': {'unit_type': 'gru',
                             'num_layers': 1,
                             'stack_width': 1500,
                             'stack_depth': 200,
                             'optimizer': 'adadelta',
                             'optimizer__global__weight_decay': 0.0000,
                             'optimizer__global__lr': 0.001},
            'expert_model_params': {'model_dir': './model_dir/expert',
                                    'd_model': 128,
                                    'rnn_num_layers': 2,
                                    'dropout': 0.8,
                                    'is_bidirectional': False,
                                    'unit_type': 'lstm'}
            }


def get_hparam_config(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IRL for Structural Evolution of Small Molecules')
    parser.add_argument('--demo', dest='demo_file', type=str,
                        help='File containing SMILES strings which are demonstrations of the required objective')
    parser.add_argument('--unbiased', dest='unbiased_file', type=str,
                        help='File containing SMILES generated with the pretrained (prior) model.')
    parser.add_argument('--prior_data', dest='prior_data', type=str,
                        help='File containing SMILES used to train the prior model.')
    parser.add_argument('--model_dir',
                        type=str,
                        default='./model_dir',
                        help='Directory containing models')
    parser.add_argument('--pretrained_model',
                        type=str,
                        default=None,
                        help='The name of the pretrained model')
    parser.add_argument("--hparam_search", action="store_true",
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
