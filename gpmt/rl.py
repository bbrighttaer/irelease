# Author: bbrighttaer
# Project: GPMT
# Date: 4/9/2020
# Time: 8:02 PM
# File: rl.py

from __future__ import absolute_import, division, print_function, unicode_literals

from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from ptan.actions import ActionSelector
from ptan.agent import BaseAgent
from tqdm import trange

from gpmt.utils import seq2tensor, get_default_tokens, pad_sequences, canonical_smiles

EpisodeStep = namedtuple('EpisodeStep', ['state', 'action'])
Trajectory = namedtuple('Trajectory', ['terminal_state', 'traj_prob'])


class MolEnvProbabilityActionSelector(ActionSelector):
    """Selects an action"""

    def __init__(self, actions):
        self.actions = actions

    def __call__(self, probs):
        assert isinstance(probs, np.ndarray)
        action_idx = np.random.choice(len(self.actions), p=probs)
        action_prob = probs[action_idx]
        action = self.actions[action_idx]
        return action, action_prob


class StateActionProbRegistry:
    """Helper class to retrieve action probabilities"""

    def __init__(self):
        self._probs_dict = {}

    def add(self, state, action, prob):
        assert isinstance(state, list) and isinstance(action, str) and isinstance(prob, float)
        self._probs_dict[(''.join(state), action)] = prob

    def get(self, state, action):
        """
        Retrieves the probability of the action in the given state.
        :param state: list
        :param action: str
        :return: float
        """
        assert isinstance(state, list) and isinstance(action, str)
        return self._probs_dict[(''.join(state), action)]

    def clear(self):
        self._probs_dict.clear()


class PolicyAgent(BaseAgent):
    def __init__(self, model, action_selector, states_preprocessor=seq2tensor, initial_state=None,
                 initial_state_args=None, apply_softmax=True, device='cpu', probs_registry=None):
        assert callable(states_preprocessor)
        if probs_registry:
            assert isinstance(probs_registry, StateActionProbRegistry)
        if initial_state:
            assert callable(initial_state)
            assert isinstance(initial_state_args, dict)
        self.model = model
        self.action_selector = action_selector
        self.states_preprocessor = states_preprocessor
        self.apply_softmax = apply_softmax
        self.device = device
        self.init_state = initial_state
        self.initial_state_args = initial_state_args
        self.probs_reg = probs_registry

    def initial_state(self):
        return self.init_state(batch_size=1, **self.initial_state_args)

    @torch.no_grad()
    def __call__(self, states, agent_states=None, **kwargs):
        """
        Selects agent actions.

        :param states: state of the environment
        :param agent_states: hidden states (in the case of RNNs)
        :return: action and agent states
        """
        if 'monte_carlo' in kwargs and kwargs['monte_carlo'] and agent_states is None:
            agent_states = [self.initial_state()]
        if agent_states is None:
            agent_states = [None] * len(states)
        state, agent_states = states[0][-1], agent_states[0]
        state, _ = self.states_preprocessor(state, self.action_selector.actions)
        state = torch.from_numpy(state).long().to(self.device)
        x = [state] + agent_states
        outputs = self.model(x)
        if isinstance(outputs, list):  # RNN case
            probs_v = outputs[0][-1]
            agent_states = outputs[1:]
        else:  # trans-decoder
            probs_v = outputs
        if self.apply_softmax:
            probs_v = torch.softmax(probs_v, dim=-1)
        probs = probs_v.data.cpu().squeeze().numpy()
        action, action_prob = self.action_selector(probs)
        if self.probs_reg and 'monte_carlo' not in kwargs:
            self.probs_reg.add(list(states[0]), action, float(action_prob))
        return action, [agent_states]


class DRLAlgorithm(object):
    """Base class for all DRL algorithms"""

    def fit(self, *args, **kwargs):
        """Implements the training procedure of the algorithm"""
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        self.fit(*args, **kwargs)


def calc_Qvals(rewards, gamma):
    qval = []
    sum_r = 0.
    for r in reversed(rewards):
        sum_r *= gamma
        sum_r += r
        qval.append(sum_r)
    return list(reversed(qval))


def unpack_batch(trajs, gamma):
    batch_states, batch_actions, batch_qvals = [], [], []
    for traj in trajs:
        rewards = []
        for exp in traj:
            batch_states.append(exp.state)
            batch_actions.append(exp.action)
            rewards.append(exp.reward)
        batch_qvals.extend(calc_Qvals(rewards, gamma))
    return batch_states, batch_actions, batch_qvals


class REINFORCE(DRLAlgorithm):
    def __init__(self, model, optimizer, initial_states_func, initial_states_args, gamma=0.97,
                 reinforce_batch=1, device='cpu'):
        assert callable(initial_states_func)
        assert isinstance(initial_states_args, dict)
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.gamma = gamma
        self.initial_states_func = initial_states_func
        self.initial_states_args = initial_states_args
        self.reinforce_batch = reinforce_batch

    @torch.enable_grad()
    def fit(self, trajectories):
        """
        Implements the REINFORCE training algorithm.

        Arguments:
        --------------
        :param states: list
            The raw states from the environment .
        :param actions: list
            The actions corresponding to each state.
        :param qvals: list
            The Q-values or Returns corresponding to each state.
        """
        states_data, actions_data, qvals = unpack_batch(trajectories, self.gamma)
        assert len(states_data) == len(actions_data) == len(qvals)
        (states_data, states_len_data), actions_data, = _preprocess_states_actions(actions_data, states_data,
                                                                                   self.device)
        self.optimizer.zero_grad()
        loss_sum = 0
        for batch_ofs in range(0, len(states_data), self.reinforce_batch):
            states = states_data[batch_ofs:batch_ofs + self.reinforce_batch]
            states_len = states_len_data[batch_ofs:batch_ofs + self.reinforce_batch]
            actions = actions_data[batch_ofs:batch_ofs + self.reinforce_batch]

            hidden_states = self.initial_states_func(states.shape[0], **self.initial_states_args)
            qvals = torch.tensor(qvals).float().to(self.device).view(-1, 1)
            outputs = self.model([states] + hidden_states)
            x = outputs[0]
            states_len = states_len - 1
            x = torch.cat([x[states_len[i], i, :].reshape(1, -1) for i in range(x.shape[1])], dim=0)
            log_probs = torch.log_softmax(x, dim=-1)
            loss = qvals * log_probs[range(qvals.shape[0]), actions]
            loss = loss.mean()
            loss_max = -loss  # for maximization since pytorch optimizers minimize by default
            loss_max.backward(retain_graph=True)
            loss_sum += loss.item()
        self.optimizer.step()
        return loss_sum


def _preprocess_states_actions(actions, states, device):
    # Process states and actions
    states = [''.join(list(state)) for state in states]
    states, states_len = pad_sequences(states)
    states, _ = seq2tensor(states, get_default_tokens())
    states = torch.from_numpy(states).long().to(device)
    states_len = torch.tensor(states_len).long().to(device)
    actions, _ = seq2tensor(actions, get_default_tokens())
    actions = torch.from_numpy(actions.reshape(-1)).long().to(device)
    return (states, states_len), actions


class PPO(DRLAlgorithm):
    """
    Proximal Policy Optimization, see: https://arxiv.org/abs/1707.06347
    Credits:
    https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter15/04_train_ppo.py
    for a good tutorial on PPO.

    Arguments:
    -----------
    :param actor:
    :param critic:
    :param actor_opt:
    :param critic_opt:
    :param initial_states_func:
    :param device:
    """

    def __init__(self, actor, critic, actor_opt, critic_opt, initial_states_func, initial_states_args, gamma=0.99,
                 gae_lambda=0.95, ppo_eps=0.2, ppo_epochs=10, ppo_batch=64, device='cpu'):
        assert callable(initial_states_func)
        assert isinstance(initial_states_args, dict)
        self.actor = actor
        self.critic = critic
        self.actor_opt = actor_opt
        self.critic_opt = critic_opt
        self.initial_states_func = initial_states_func
        self.initial_states_args = initial_states_args
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_eps = ppo_eps
        self.ppo_epochs = ppo_epochs
        self.ppo_batch = ppo_batch

    @property
    def model(self):
        return self.actor

    def calc_adv_ref(self, trajectory):
        states, actions, _ = unpack_batch([trajectory], self.gamma)
        last_state = ''.join(list(states[-1]))
        inp, _ = seq2tensor([last_state], tokens=get_default_tokens())
        inp = torch.from_numpy(inp).long().to(self.device)
        values_v = self.critic(inp)
        values = values_v.view(-1, ).data.cpu().numpy()
        last_gae = 0.0
        result_adv = []
        result_ref = []
        for val, next_val, exp in zip(reversed(values[:-1]), reversed(values[1:]), reversed(trajectory[:-1])):
            if exp.last_state is None:  # for terminal state
                delta = exp.reward - val
                last_gae = delta
            else:
                delta = exp.reward + self.gamma * next_val - val
                last_gae = delta + self.gamma * self.gae_lambda * last_gae
            result_adv.append(last_gae)
            result_ref.append(last_gae + val)

        adv_v = torch.FloatTensor(list(reversed(result_adv))).to(self.device)
        ref_v = torch.FloatTensor(list(reversed(result_ref))).to(self.device)
        return states[:-1], actions[:-1], adv_v, ref_v

    @torch.enable_grad()
    def fit(self, trajectories):
        # Calculate GAE
        batch_states, batch_actions, batch_adv, batch_ref = [], [], [], []
        for traj in trajectories:
            states, actions, adv_v, ref_v = self.calc_adv_ref(traj)
            batch_states.extend(states)
            batch_actions.extend(actions)
            batch_adv.extend(adv_v)
            batch_ref.extend(ref_v)

        # Normalize advantages
        batch_ref = torch.tensor(batch_ref).float().to(self.device)
        batch_adv = torch.tensor(batch_adv).float().to(self.device)
        batch_adv = (batch_adv - batch_adv.mean()) / batch_adv.std()

        # Calculate old probs of actions
        (states, states_len), actions, = _preprocess_states_actions(batch_actions, batch_states, self.device)
        hidden_states = self.initial_states_func(batch_size=states.shape[0], **self.initial_states_args)
        with torch.set_grad_enabled(False):
            outputs = self.actor([states] + hidden_states)
        x = outputs[0]
        states_len = states_len - 1  # to select actions since samples are padded
        x = torch.cat([x[states_len[i], i, :].reshape(1, -1) for i in range(x.shape[1])], dim=0).to(self.device)
        old_log_probs = torch.log_softmax(x, dim=-1).detach()
        old_log_probs = old_log_probs[range(old_log_probs.shape[0]), actions]

        sum_loss_value = []
        sum_loss_policy = []

        for epoch in trange(self.ppo_epochs, desc='PPO optimization...'):
            for batch_ofs in range(0, len(batch_states), self.ppo_batch):
                # Select batch data
                states_v = states[batch_ofs:batch_ofs + self.ppo_batch]
                states_len_v = states_len[batch_ofs:batch_ofs + self.ppo_batch]
                actions_v = actions[batch_ofs:batch_ofs + self.ppo_batch]
                batch_adv_v = batch_adv[batch_ofs:batch_ofs + self.ppo_batch]
                batch_ref_v = batch_ref[batch_ofs:batch_ofs + self.ppo_batch]
                old_log_probs_v = old_log_probs[batch_ofs:batch_ofs + self.ppo_batch]
                hidden_states_v = self.initial_states_func(batch_size=states_v.shape[0], **self.initial_states_args)

                # Critic training
                self.critic_opt.zero_grad()
                value_v = self.critic(states_v)
                value_v = torch.cat([value_v[states_len_v[i], i, :] for i in range(value_v.shape[1])], dim=0)
                value_v = value_v.to(self.device)
                loss_value_v = F.mse_loss(value_v, batch_ref_v)
                loss_value_v.backward()
                self.critic_opt.step()
                sum_loss_value.append(loss_value_v.item())

                # Actor training
                self.actor_opt.zero_grad()
                x_v = self.actor([states_v] + hidden_states_v)
                x_v = x_v[0]
                x_v = torch.cat([x_v[states_len_v[i], i, :].reshape(1, -1) for i in range(x_v.shape[1])], dim=0)
                logprob_pi_v = torch.log_softmax(x_v, dim=-1)
                logprob_pi_v = logprob_pi_v[range(logprob_pi_v.shape[0]), actions_v]
                ratio_v = torch.exp(logprob_pi_v - old_log_probs_v)
                surr_obj_v = batch_adv_v * ratio_v
                clipped_surr_v = batch_adv_v * torch.clamp(ratio_v, 1.0 - self.ppo_eps, 1.0 + self.ppo_eps)
                loss_policy_v = torch.min(surr_obj_v, clipped_surr_v).mean()
                sum_loss_policy.append(loss_policy_v.item())
                loss_policy_v = -loss_policy_v  # for maximization
                loss_policy_v.backward()
                self.actor_opt.step()
        return np.mean(sum_loss_value), np.mean(sum_loss_policy)


class GuidedRewardLearningIRL(DRLAlgorithm):
    """
    Implementation of:
    “Guided Cost Learning : Deep Inverse Optimal Control via Policy Optimization,” vol. 48, 2016.
    """

    def __init__(self, model, optimizer, demo_gen_data, agent_net, agent_net_init_func, agent_net_init_func_args, k=10,
                 use_buffer=True, buffer_size=1000, buffer_batch_size=100, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.demo_gen_data = demo_gen_data
        self.k = k
        self.device = device
        self.use_buffer = use_buffer
        if use_buffer:
            self.replay_buffer = TrajectoriesBuffer(buffer_size)
        else:
            self.replay_buffer = None
        self.batch_size = buffer_batch_size
        self.agent_net = agent_net
        self.agent_net_init_func = agent_net_init_func
        self.agent_net_init_args = agent_net_init_func_args

    @property
    def generator(self):
        return self.demo_gen_data

    @torch.enable_grad()
    def fit(self, trajectories):
        """Train the reward function / model using the GRL algorithm."""
        if self.use_buffer:
            extra_trajs = self.replay_buffer.sample(self.batch_size)
            trajectories.extend(extra_trajs)
            self.replay_buffer.populate(trajectories)
        d_traj, d_traj_probs = [], []
        for traj in trajectories:
            d_traj.append(''.join(list(traj.terminal_state.state)) + traj.terminal_state.action)
            d_traj_probs.append(traj.traj_prob)
        _, valid_vec_samp = canonical_smiles(d_traj)
        valid_vec_samp = torch.tensor(valid_vec_samp).view(-1, 1).float().to(self.device)
        d_traj, _ = pad_sequences(d_traj)
        d_samp, _ = seq2tensor(d_traj, tokens=get_default_tokens())
        d_samp = torch.from_numpy(d_samp).long().to(self.device)
        losses = []
        pbar = trange(self.k, desc='IRL optimization...')
        for i in pbar:
            # D_demo processing
            demo_states, demo_actions = self.demo_gen_data.random_training_set()
            d_demo = torch.cat([demo_states, demo_actions[:, -1].reshape(-1, 1)], dim=1)
            valid_vec_demo = torch.ones(d_demo.shape[0]).view(-1, 1).float().to(self.device)
            d_demo_out = self.model([d_demo, valid_vec_demo])

            # D_samp processing
            d_samp_out = self.model([d_samp, valid_vec_samp])
            d_out_combined = torch.cat([d_samp_out, d_demo_out], dim=0)
            if d_samp_out.shape[0] < 1000:
                d_samp_out = d_out_combined
            z = torch.ones(d_samp_out.shape[0]).float().to(self.device)  # dummy importance weights TODO: replace this
            d_samp_out = z.view(-1, 1) * torch.exp(d_samp_out)

            # Get rep vectors of trajs from agent_net and calculate the similarity regularization
            pad_size = max(d_samp.shape[1], d_demo.shape[1])
            d_samp_padded = torch.cat([d_samp, torch.zeros(d_samp.shape[0], pad_size - d_samp.shape[1])
                                      .fill_(get_default_tokens().index(' ')).long().to(self.device)], dim=1)
            d_demo_padded = torch.cat([d_demo, torch.zeros(d_demo.shape[0], pad_size - d_demo.shape[1])
                                      .fill_(get_default_tokens().index(' ')).long().to(self.device)], dim=1)
            d_samp_demo = torch.cat([d_samp_padded, d_demo_padded])
            hidden = self.agent_net_init_func(d_samp_demo.shape[0], **self.agent_net_init_args)
            with torch.set_grad_enabled(False):
                outputs = self.agent_net([d_samp_demo] + hidden)
            reps = outputs[0].detach()  # shape structure: (seq. len, batch, d_model)
            reps = reps[-1, :, :]  # select last rep
            reps = reps / torch.norm(reps, dim=-1, keepdim=True)  # normalize vectors
            sim_values = reps @ reps.t()  # cosine similarity
            xx, yy = torch.meshgrid(d_out_combined.view(-1, ), d_out_combined.view(-1, ))
            sqr_diffs = torch.pow(xx - yy, 2)
            reg_loss = sim_values * sqr_diffs
            reg_loss = torch.mean(torch.sum(reg_loss, dim=1))

            # objective
            loss = torch.mean(d_demo_out) - torch.log(torch.mean(d_samp_out)) - reg_loss
            losses.append(loss.item())
            loss = -loss  # for maximization

            # update params
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return np.mean(losses)


class TrajectoriesBuffer:
    """
    Stores trajectories generated by different background distributions
    adapted from ptan.ExperienceReplayBuffer
    """

    def __init__(self, buffer_size):
        self.capacity = buffer_size
        self.buffer = []
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def sample(self, batch_size):
        if len(self.buffer) <= batch_size:
            return self.buffer
        keys = np.random.choice(len(self.buffer), batch_size, replace=True)
        return [self.buffer[key] for key in keys]

    def _add(self, trajectory):
        assert isinstance(trajectory, Trajectory)
        if len(self.buffer) < self.capacity:
            self.buffer.append(trajectory)
        else:
            self.buffer[self.pos] = trajectory
        self.pos = (self.pos + 1) % self.capacity

    def populate(self, samples):
        for entry in samples:
            self._add(entry)
