# Author: bbrighttaer
# Project: GPMT
# Date: 4/9/2020
# Time: 8:02 PM
# File: rl.py

from __future__ import absolute_import, division, print_function, unicode_literals

from collections import namedtuple

import numpy as np
import torch
from ptan.actions import ActionSelector
from ptan.agent import BaseAgent

from gpmt.utils import seq2tensor, get_default_tokens, pad_sequences

EpisodeStep = namedtuple('EpisodeStep', ['state', 'action', 'qval'])
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
    def __init__(self, model, action_selector, states_preprocessor=seq2tensor, initial_state=None, apply_softmax=True,
                 device='cpu', probs_registry=None):
        assert callable(states_preprocessor)
        if probs_registry:
            assert isinstance(probs_registry, StateActionProbRegistry)
        if initial_state:
            assert callable(initial_state)
        self.model = model
        self.action_selector = action_selector
        self.states_preprocessor = states_preprocessor
        self.apply_softmax = apply_softmax
        self.device = device
        self.init_state = initial_state
        self.probs_reg = probs_registry

    def initial_state(self):
        return self.init_state()

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


class REINFORCE(DRLAlgorithm):
    def __init__(self, model, optimizer, initial_states_func, device='cpu'):
        assert callable(initial_states_func)
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.initial_states_func = initial_states_func

    @torch.enable_grad()
    def fit(self, states, actions, qvals):
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
        assert len(states) == len(actions) == len(qvals)
        (states, states_len), actions, = _preprocess_states_actions(actions, states, self.device)
        hidden_states = self.initial_states_func(states.shape[0])
        qvals = torch.tensor(qvals).float().to(self.device).view(-1, 1)
        self.optimizer.zero_grad()
        outputs = self.model([states] + hidden_states)
        x = outputs[0]
        states_len = states_len - 1
        x = torch.cat([x[states_len[i], i, :].reshape(1, -1) for i in range(x.shape[1])], dim=0)
        log_probs = torch.log_softmax(x, dim=-1)
        loss = qvals * log_probs[range(qvals.shape[0]), actions]
        loss = -loss.mean()  # for maximization since pytorch optimizers minimize by default
        loss.backward()
        self.optimizer.step()
        return -loss.item()


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


class GuidedRewardLearningIRL(DRLAlgorithm):
    """
    Implementation of:
    “Guided Cost Learning : Deep Inverse Optimal Control via Policy Optimization,” vol. 48, 2016.
    """

    def __init__(self, model, optimizer, demo_gen_data, k=10, use_buffer=True, buffer_size=1000, buffer_batch_size=100,
                 device='cpu'):
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
        d_traj, _ = pad_sequences(d_traj)
        d_samp, _ = seq2tensor(d_traj, tokens=get_default_tokens())
        losses = []
        for i in range(self.k):
            demo_states, demo_actions = self.demo_gen_data.random_training_set()
            d_demo_trajs = torch.cat([demo_states, demo_actions[:, -1].reshape(-1, 1)], dim=1)

        return losses


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
