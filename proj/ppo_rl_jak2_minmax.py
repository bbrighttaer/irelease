# Author: bbrighttaer
# Project: GPMT
# Date: 4/8/2020
# Time: 8:02 PM
# File: ppo_rl_jak2_minmax.py

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import contextlib
import copy
import math
import os
import random
import time
from datetime import datetime as dt

import numpy as np
import torch
import torch.nn as nn
from ptan.common.utils import TBMeanTracker
from ptan.experience import ExperienceSourceFirstLast
from soek import Trainer, DataNode, RandomSearch, BayesianOptSearch, ConstantParam, RealParam, DiscreteParam, DictParam, \
    CategoricalParam, LogRealParam
from soek.bopt import GPMinArgs
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from irelease.data import GeneratorData
from irelease.env import MoleculeEnv
from irelease.model import Encoder, StackRNN, RNNLinearOut, \
    CriticRNN, RewardNetRNN, StackedRNNDropout, StackedRNNLayerNorm
from irelease.mol_metrics import verify_sequence, get_mol_metrics
from irelease.predictor import get_jak2_max_reward, get_jak2_min_reward, XGBPredictor, get_jak2_max_baseline_reward, \
    get_jak2_min_baseline_reward
from irelease.reward import RewardFunction
from irelease.rl import MolEnvProbabilityActionSelector, PolicyAgent, GuidedRewardLearningIRL, \
    StateActionProbRegistry, Trajectory, EpisodeStep, PPO
from irelease.utils import Flags, get_default_tokens, parse_optimizer, seq2tensor, init_hidden, init_cell, init_stack, \
    time_since, generate_smiles, ExpAverage

currentDT = dt.now()
date_label = currentDT.strftime("%Y_%m_%d__%H_%M_%S")
seeds = [0]

if torch.cuda.is_available():
    dvc_id = 1
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
            if hparams['agent_params']['num_layers'] > 1:
                rnn_layers.append(StackedRNNDropout(hparams['dropout']))
                rnn_layers.append(StackedRNNLayerNorm(hparams['d_model']))
        agent_net = nn.Sequential(encoder,
                                  *rnn_layers,
                                  RNNLinearOut(out_dim=demo_data_gen.n_characters,
                                               hidden_size=hparams['d_model'],
                                               bidirectional=False,
                                               bias=True))
        optimizer_agent_net = parse_optimizer(hparams['agent_params'], agent_net)
        with contextlib.suppress(Exception):
            agent_net = agent_net.to(device)
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
        critic = nn.Sequential(encoder,
                               CriticRNN(hparams['d_model'], hparams['critic_params']['d_model'],
                                         unit_type=hparams['critic_params']['unit_type'],
                                         num_layers=hparams['critic_params']['num_layers']))
        optimizer_critic_net = parse_optimizer(hparams['critic_params'], critic)
        with contextlib.suppress(Exception):
            critic = critic.to(device)
        drl_alg = PPO(actor=agent_net, actor_opt=optimizer_agent_net,
                      critic=critic, critic_opt=optimizer_critic_net,
                      initial_states_func=agent_net_hidden_states_func,
                      initial_states_args=init_state_args,
                      device=device,
                      gamma=hparams['gamma'],
                      gae_lambda=hparams['gae_lambda'],
                      ppo_eps=hparams['ppo_eps'],
                      ppo_epochs=hparams['ppo_epochs'],
                      ppo_batch=hparams['ppo_batch'],
                      entropy_beta=hparams['entropy_beta'])

        # Reward function entities
        reward_net = nn.Sequential(encoder,
                                   RewardNetRNN(input_size=hparams['d_model'],
                                                hidden_size=hparams['reward_params']['d_model'],
                                                num_layers=hparams['reward_params']['num_layers'],
                                                bidirectional=hparams['reward_params']['bidirectional'],
                                                use_attention=hparams['reward_params']['use_attention'],
                                                dropout=hparams['dropout'],
                                                unit_type=hparams['reward_params']['unit_type'],
                                                use_smiles_validity_flag=hparams['reward_params']['use_validity_flag']))
        optimizer_reward_net = parse_optimizer(hparams['reward_params'], reward_net)
        with contextlib.suppress(Exception):
            reward_net = reward_net.to(device)

        expert_model = XGBPredictor(hparams['expert_model_dir'])
        if hparams['baseline_reward']:
            true_reward_func = get_jak2_max_baseline_reward if hparams['bias_mode'] == 'max' \
                else get_jak2_min_baseline_reward
        else:
            true_reward_func = get_jak2_max_reward if hparams['bias_mode'] == 'max' else get_jak2_min_reward
        reward_function = RewardFunction(reward_net, mc_policy=agent, actions=demo_data_gen.all_characters,
                                         device=device, use_mc=hparams['use_monte_carlo_sim'],
                                         mc_max_sims=hparams['monte_carlo_N'],
                                         use_true_reward=hparams['use_true_reward'],
                                         true_reward_func=true_reward_func,
                                         expert_func=expert_model,
                                         no_mc_fill_val=hparams['no_mc_fill_val'])
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
    def data_provider(k, flags, seed):
        tokens = get_default_tokens()
        demo_data = GeneratorData(training_data_path=flags.demo_file,
                                  delimiter='\t',
                                  cols_to_read=[0],
                                  keep_header=True,
                                  pad_symbol=' ',
                                  max_len=120,
                                  tokens=tokens,
                                  seed=seed,
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
    def evaluate(res_dict, generated_smiles, ref_smiles):
        smiles = []
        for s in generated_smiles:
            if verify_sequence(s):
                smiles.append(s)
        mol_metrics = get_mol_metrics()
        for metric in mol_metrics:
            res_dict[metric] = np.mean(mol_metrics[metric](smiles, ref_smiles))
        score = res_dict['internal_diversity']
        return score

    @staticmethod
    def train(init_args, bias_mode, agent_net_path=None, agent_net_name=None, n_episodes=500,
              sim_data_node=None, tb_writer=None, is_hsearch=False, n_to_generate=200, learn_irl=True):
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
        exp_avg = ExpAverage(beta=0.6)
        mean_preds_exp_avg = ExpAverage(beta=0.6)
        best_score = -1.

        # load pretrained model
        if agent_net_path and agent_net_name:
            try:
                print('Loading pretrained model...')
                weights = IReLeaSE.load_model(agent_net_path, agent_net_name)
                agent.model.load_state_dict(weights)
                print('Pretrained model loaded successfully!')
            except:
                print('Pretrained model could not be loaded. Terminating prematurely.')
                return {'model': [drl_algorithm.actor,
                                  drl_algorithm.critic,
                                  irl_algorithm.model],
                        'score': round(best_score, 3),
                        'epoch': -1}

        start = time.time()

        # Begin simulation and training
        total_rewards = []
        trajectories = []
        done_episodes = 0
        batch_episodes = 0
        exp_trajectories = []
        step_idx = 0

        # collect mean predictions
        unbiased_smiles_mean_pred, biased_smiles_mean_pred, gen_smiles_mean_pred = [], [], []
        unbiased_smiles_mean_pred_data_node = DataNode('baseline_mean_vals', unbiased_smiles_mean_pred)
        biased_smiles_mean_pred_data_node = DataNode('biased_mean_vals', biased_smiles_mean_pred)
        gen_smiles_mean_pred_data_node = DataNode('gen_mean_vals', gen_smiles_mean_pred)
        if sim_data_node:
            sim_data_node.data = [unbiased_smiles_mean_pred_data_node,
                                  biased_smiles_mean_pred_data_node,
                                  gen_smiles_mean_pred_data_node]

        env = MoleculeEnv(actions=get_default_tokens(), reward_func=reward_func)
        exp_source = ExperienceSourceFirstLast(env, agent, gamma, steps_count=1, steps_delta=1)
        traj_prob = 1.
        exp_traj = []

        demo_score = np.mean(expert_model(demo_data_gen.random_training_set_smiles(1000))[1])
        baseline_score = np.mean(expert_model(unbiased_data_gen.random_training_set_smiles(1000))[1])
        # with contextlib.suppress(RuntimeError if is_hsearch else DummyException):
        try:
            with TBMeanTracker(tb_writer, 1) as tracker:
                for step_idx, exp in tqdm(enumerate(exp_source)):
                    exp_traj.append(exp)
                    traj_prob *= probs_reg.get(list(exp.state), exp.action)

                    if exp.last_state is None:
                        trajectories.append(Trajectory(terminal_state=EpisodeStep(exp.state, exp.action),
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
                        mean_preds = np.nanmean(predictions)
                        mean_preds_exp_avg.update(mean_preds)
                        if math.isnan(mean_preds) or math.isinf(mean_preds):
                            print(f'mean preds is {mean_preds}, terminating')
                            # best_score = -1.
                            break
                        try:
                            if bias_mode == 'max':
                                percentage_in_threshold = np.sum((predictions >= demo_score)) / len(predictions)
                            else:
                                percentage_in_threshold = np.sum((predictions <= demo_score)) / len(predictions)
                        except:
                            percentage_in_threshold = 0.
                        per_valid = len(predictions) / n_to_generate
                        if per_valid < 0.2:
                            print(f'Percentage of valid SMILES is = {per_valid}. Terminating...')
                            # best_score = -1.
                            break
                        print(f'Mean value of predictions = {mean_preds}, % of valid SMILES = {per_valid}')
                        unbiased_smiles_mean_pred.append(float(baseline_score))
                        biased_smiles_mean_pred.append(float(demo_score))
                        gen_smiles_mean_pred.append(float(mean_preds))
                        tb_writer.add_scalars('qsar_score', {'sampled': mean_preds,
                                                             'baseline': baseline_score,
                                                             'demo_data': demo_score}, step_idx)
                        tb_writer.add_scalars('SMILES stats', {'per. of valid': per_valid,
                                                               'per. in drug-like region': percentage_in_threshold},
                                              step_idx)
                        eval_dict = {}
                        eval_score = IReLeaSE.evaluate(eval_dict, samples,
                                                       demo_data_gen.random_training_set_smiles(1000))
                        for k in eval_dict:
                            tracker.track(k, eval_dict[k], step_idx)
                        avg_len = np.nanmean([len(s) for s in samples])
                        tracker.track('Average SMILES length', np.nanmean([len(s) for s in samples]), step_idx)
                        d_penalty = eval_score < .5
                        s_penalty = avg_len < 20
                        if bias_mode == 'max':
                            diff = mean_preds - demo_score
                            stop_condition = mean_preds_exp_avg.value >= demo_score
                        else:
                            diff = demo_score - mean_preds
                            stop_condition = mean_preds_exp_avg.value <= demo_score
                        # score = 3 * np.exp(diff) + np.log(per_valid + 1e-5) - s_penalty * np.exp(
                        #     diff) - d_penalty * np.exp(diff)
                        score = np.exp(diff)
                        # score = np.exp(diff) + np.mean([np.exp(per_valid), np.exp(percentage_in_threshold)])
                        if math.isnan(score) or math.isinf(score):
                            # best_score = -1.
                            print(f'Score is {score}, terminating.')
                            break
                        tracker.track('score', score, step_idx)
                        exp_avg.update(score)
                        if is_hsearch:
                            best_score = exp_avg.value
                        if exp_avg.value > best_score:
                            best_model_wts = [copy.deepcopy(drl_algorithm.actor.state_dict()),
                                              copy.deepcopy(drl_algorithm.critic.state_dict()),
                                              copy.deepcopy(irl_algorithm.model.state_dict())]
                            best_score = exp_avg.value
                        if stop_condition:
                            print(f'threshold reached, best score={mean_preds_exp_avg.value}, '
                                  f'threshold={demo_score}, training completed')
                            break
                        if done_episodes == n_episodes or stop_condition:
                            print('Training completed!')
                            break

                    if batch_episodes < episodes_to_train:
                        continue

                    # Train models
                    print('Fitting models...')
                    irl_loss = irl_algorithm.fit(trajectories) if learn_irl else 0.
                    rl_loss = drl_algorithm.fit(exp_trajectories)
                    samples = generate_smiles(drl_algorithm.model, demo_data_gen, init_args['gen_args'],
                                              num_samples=3)
                    print(f'IRL loss = {irl_loss}, RL loss = {rl_loss}, samples = {samples}')
                    tracker.track('irl_loss', irl_loss, step_idx)
                    tracker.track('critic_loss', rl_loss[0], step_idx)
                    tracker.track('agent_loss', rl_loss[1], step_idx)

                    # Reset
                    batch_episodes = 0
                    trajectories.clear()
                    exp_trajectories.clear()
        except FileNotFoundError as e:
            print(str(e))
        if best_model_wts:
            drl_algorithm.actor.load_state_dict(best_model_wts[0])
            drl_algorithm.critic.load_state_dict(best_model_wts[1])
            irl_algorithm.model.load_state_dict(best_model_wts[2])
        duration = time.time() - start
        print('\nTraining duration: {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))
        # if math.isinf(best_score) or math.isnan(best_score):
        #     best_score = -1.
        return {'model': [drl_algorithm.actor,
                          drl_algorithm.critic,
                          irl_algorithm.model],
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
    if flags.use_true_reward:
        reward_label = '_baseline_reward' if flags.baseline_reward else '_true_reward'
    else:
        reward_label = ''
    sim_label = flags.exp_name + '_IReLeaSE-PPO_' + (
        reward_label if flags.use_true_reward else 'with_irl') + ('_no_vflag' if flags.no_smiles_validity_flag else '')
    sim_data = DataNode(label=sim_label, metadata={'exp': flags.exp_name, 'date': date_label})
    nodes_list = []
    sim_data.data = nodes_list

    for seed in seeds:
        summary_writer_creator = lambda: SummaryWriter(log_dir="irelease_tb"
                                                               "/{}_{}_{}/".format(sim_label, seed, dt.now().strftime(
            "%Y_%m_%d__%H_%M_%S")))

        # for data collection of this round of simulation.
        data_node = DataNode(label="seed_%d" % seed)
        nodes_list.append(data_node)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        flags['seed'] = seed

        print('--------------------------------------------------------------------------------')
        print(f'{device}\n{sim_label}\tDemonstrations file: {flags.demo_file}')
        print('--------------------------------------------------------------------------------')

        irelease = IReLeaSE()
        k = 1
        if flags.hparam_search:
            try:
                print(f'Hyperparameter search enabled: {flags.hparam_search_alg}')
                # arguments to callables
                extra_init_args = {}
                extra_data_args = {'flags': flags, 'seed': seed}
                extra_train_args = {'agent_net_path': flags.model_dir,
                                    'agent_net_name': flags.pretrained_model,
                                    'learn_irl': not flags.use_true_reward,
                                    'bias_mode': flags.bias_mode,
                                    'n_episodes': 110,
                                    'is_hsearch': True,
                                    'tb_writer': summary_writer_creator}
                hparams_conf = get_hparam_config(flags)
                search_alg = {'random_search': RandomSearch,
                              'bayopt_search': BayesianOptSearch}.get(flags.hparam_search_alg,
                                                                      BayesianOptSearch)
                search_args = GPMinArgs(n_calls=20, random_state=seed)
                # search_args = GBRTMinArgs(n_calls=20, random_state=seed)
                hparam_search = search_alg(hparam_config=hparams_conf,
                                           num_folds=1,
                                           initializer=irelease.initialize,
                                           data_provider=irelease.data_provider,
                                           train_fn=irelease.train,
                                           save_model_fn=irelease.save_model,
                                           alg_args=search_args,
                                           init_args=extra_init_args,
                                           data_args=extra_data_args,
                                           train_args=extra_train_args,
                                           data_node=data_node,
                                           split_label='ppo-rl',
                                           sim_label=sim_label,
                                           dataset_label=None,
                                           results_file=f'{flags.hparam_search_alg}_{sim_label}'
                                                        f'_{date_label}_seed{seed}')
                start = time.time()
                stats = hparam_search.fit()
                print(f'Duration = {time_since(start)}')
                print(stats)
                print("\nBest params = {}, duration={}".format(stats.best(), time_since(start)))
            except ValueError as e:
                print(str(e))

        else:
            if flags.bias_mode == 'min':
                hyper_params = default_hparams_min(flags)
            else:
                hyper_params = default_hparams_max(flags)
            # hyper_params = parse_hparams(
            #     'bayopt_search_JAK2_max_IReLeaSE-ppo_with_irl_no_attn_2020_06_23__06_22_11_seed0_gp.csv', 1)
            # print(f'params={hyper_params}')
            data_gens = irelease.data_provider(k, flags, seed)
            init_args = irelease.initialize(hyper_params, data_gens['demo_data'], data_gens['unbiased_data'],
                                            data_gens['prior_data'])
            results = irelease.train(init_args, flags.bias_mode, flags.model_dir, flags.pretrained_model,
                                     sim_data_node=data_node,
                                     n_episodes=100,
                                     learn_irl=not flags.use_true_reward,
                                     tb_writer=summary_writer_creator)
            irelease.save_model(results['model'][0],
                                path=flags.model_dir,
                                name=f'{flags.exp_name}_{flags.bias_mode}_irelease_stack-rnn_'
                                     f'{hyper_params["agent_params"]["unit_type"]}'
                                     f'_ppo_agent{reward_label}_{date_label}_{results["score"]}_{results["epoch"]}')
            irelease.save_model(results['model'][1],
                                path=flags.model_dir,
                                name=f'{flags.exp_name}_{flags.bias_mode}_irelease_stack-rnn_'
                                     f'{hyper_params["agent_params"]["unit_type"]}'
                                     f'_ppo_critic{reward_label}_{date_label}_{results["score"]}_{results["epoch"]}')
            irelease.save_model(results['model'][2],
                                path=flags.model_dir,
                                name=f'{flags.exp_name}_{flags.bias_mode}_irelease_stack-rnn_'
                                     f'{hyper_params["agent_params"]["unit_type"]}'
                                     f'_ppo_reward_net{reward_label}_{date_label}_{results["score"]}_{results["epoch"]}')

    # save simulation data resource tree to file.
    sim_data.to_json(path="./analysis/")


def default_hparams_min(args):
    return {'d_model': 1500,
            'dropout': 0.0,
            'monte_carlo_N': 5,
            'use_monte_carlo_sim': True,
            'no_mc_fill_val': 0.0,
            'gamma': 0.97,
            'episodes_to_train': 6,
            'gae_lambda': 0.95,
            'ppo_eps': 0.2,
            'ppo_batch': 1,
            'ppo_epochs': 3,
            'entropy_beta': 0.05,
            'bias_mode': args.bias_mode,
            'use_true_reward': args.use_true_reward,
            'baseline_reward': args.baseline_reward,
            'reward_params': {'num_layers': 2,
                              'd_model': 512,
                              'unit_type': 'gru',
                              'demo_batch_size': 128,
                              'irl_alg_num_iter': 4,
                              'dropout': 0.0,
                              'use_attention': args.use_attention,
                              'use_validity_flag': ~args.no_smiles_validity_flag,
                              'bidirectional': True,
                              'optimizer': 'adadelta',
                              'optimizer__global__weight_decay': 0.005,
                              'optimizer__global__lr': 0.001, },
            'agent_params': {'unit_type': 'gru',
                             'num_layers': 2,
                             'stack_width': 1500,
                             'stack_depth': 200,
                             'optimizer': 'adadelta',
                             'optimizer__global__weight_decay': 0.005,
                             'optimizer__global__lr': 0.001},
            'critic_params': {'num_layers': 2,
                              'd_model': 300,
                              'dropout': 0.2,
                              'unit_type': 'gru',
                              'optimizer': 'adam',
                              'optimizer__global__weight_decay': 0.0005,
                              'optimizer__global__lr': 0.001},
            'expert_model_dir': './model_dir/expert_xgb_reg'
            }


def default_hparams_max(args):
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
            'bias_mode': args.bias_mode,
            'use_true_reward': args.use_true_reward,
            'baseline_reward': args.baseline_reward,
            'reward_params': {'num_layers': 2,
                              'd_model': 512,
                              'unit_type': 'gru',
                              'demo_batch_size': 32,
                              'irl_alg_num_iter': 5,
                              'dropout': 0.2,
                              'use_attention': args.use_attention,
                              'use_validity_flag': not args.no_smiles_validity_flag,
                              'bidirectional': True,
                              'optimizer': 'adadelta',
                              'optimizer__global__weight_decay': 0.0005,
                              'optimizer__global__lr': 0.001, },
            'agent_params': {'unit_type': 'gru',
                             'num_layers': 2,
                             'stack_width': 1500,
                             'stack_depth': 200,
                             'optimizer': 'adadelta',
                             'optimizer__global__weight_decay': 0.005,
                             'optimizer__global__lr': 0.001},
            'critic_params': {'num_layers': 2,
                              'd_model': 256,
                              'dropout': 0.2,
                              'unit_type': 'gru',
                              'optimizer': 'adam',
                              'optimizer__global__weight_decay': 0.005,
                              'optimizer__global__lr': 0.001},
            'expert_model_dir': './model_dir/expert_xgb_reg'
            }


def get_hparam_config(args):
    return {'seed': ConstantParam(args.seed),
            'd_model': ConstantParam(1500),
            'dropout': RealParam(),
            'monte_carlo_N': ConstantParam(5),
            'use_monte_carlo_sim': ConstantParam(True),
            'no_mc_fill_val': ConstantParam(0.0),
            'gamma': ConstantParam(0.97),
            'episodes_to_train': ConstantParam(6),
            'gae_lambda': RealParam(0.9, max=0.999),
            'ppo_eps': ConstantParam(0.2),
            'ppo_batch': ConstantParam(1),
            'ppo_epochs': ConstantParam(3),
            'entropy_beta': ConstantParam(0.05),
            'bias_mode': ConstantParam(args.bias_mode),
            'use_true_reward': ConstantParam(args.use_true_reward),
            'baseline_reward': ConstantParam(args.baseline_reward),
            'reward_params': DictParam({'num_layers': ConstantParam(2),
                                        'd_model': DiscreteParam(min=128, max=500),
                                        'unit_type': ConstantParam('gru'),
                                        'demo_batch_size': ConstantParam(128),
                                        'irl_alg_num_iter': DiscreteParam(2, max=10),
                                        'use_attention': ConstantParam(False),
                                        'bidirectional': ConstantParam(True),
                                        'dropout': RealParam(),
                                        'use_validity_flag': ConstantParam(not args.no_smiles_validity_flag),
                                        'optimizer': CategoricalParam(
                                            choices=['sgd', 'adam', 'adadelta', 'adagrad', 'adamax', 'rmsprop']),
                                        'optimizer__global__weight_decay': LogRealParam(),
                                        'optimizer__global__lr': LogRealParam()}),
            'agent_params': DictParam({'unit_type': ConstantParam('gru'),
                                       'num_layers': ConstantParam(2),
                                       'stack_width': ConstantParam(1500),
                                       'stack_depth': ConstantParam(200),
                                       'optimizer': ConstantParam('adadelta'),
                                       'optimizer__global__weight_decay': ConstantParam(0.005),
                                       'optimizer__global__lr': ConstantParam(0.001)}),
            'critic_params': DictParam({'num_layers': ConstantParam(2),
                                        'd_model': ConstantParam(128),
                                        'unit_type': ConstantParam('gru'),
                                        'optimizer': ConstantParam('adam'),
                                        'optimizer__global__weight_decay': ConstantParam(0.0005),
                                        'optimizer__global__lr': ConstantParam(0.001)}),
            'expert_model_dir': ConstantParam('./model_dir/expert_xgb_reg')
            }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IRL for Structural Evolution of Small Molecules')
    parser.add_argument('--exp_name', type=str,
                        help='Name for the experiment. This would be added to saved model names')
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
    parser.add_argument('--use_attention',
                        action='store_true',
                        help='Whether to use additive attention')
    parser.add_argument('--expert_dir', type=str, default='./model_dir/expert_svr',
                        help='The directory containing SVR model(s) to predict JAK2 pIC50')
    parser.add_argument('--bias_mode', type=str, choices=['min', 'max'], default='max',
                        help='The generator biasing objective')
    parser.add_argument('--use_true_reward',
                        action='store_true',
                        help='If true then no reward function would be learned but the true reward would be used.'
                             'This requires that the explicit reward function is given.')
    parser.add_argument('--baseline_reward', action='store_true',
                        help='If true reward is enabled, this indicates whether the baseline reward option is used.')
    parser.add_argument('--no_smiles_validity_flag', action='store_true',
                        help='If True, smiles validity flag would not be passed to the reward net')

    args = parser.parse_args()
    flags = Flags()
    args_dict = args.__dict__
    for arg in args_dict:
        setattr(flags, arg, args_dict[arg])
    main(flags)
