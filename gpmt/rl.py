# Author: bbrighttaer
# Project: GPMT
# Date: 4/9/2020
# Time: 8:02 PM
# File: rl.py

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import torch
from ptan.actions import ActionSelector
from ptan.agent import BaseAgent

from gpmt.utils import seq2tensor


class MolEnvProbabilityActionSelector(ActionSelector):
    """Selects an action"""

    def __init__(self, actions):
        self.actions = actions

    def __call__(self, probs):
        assert isinstance(probs, np.ndarray)
        action = self.actions[np.random.choice(len(self.actions), p=probs)]
        return action


class PolicyAgent(BaseAgent):
    def __init__(self, model, action_selector, states_preprocessor=seq2tensor, initial_state=None, apply_softmax=True,
                 device='cpu'):
        assert callable(states_preprocessor)
        if initial_state:
            assert callable(initial_state)
        self.model = model
        self.action_selector = action_selector
        self.states_preprocessor = states_preprocessor
        self.apply_softmax = apply_softmax
        self.device = device
        self.init_state = initial_state

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
        action = self.action_selector(probs)
        return action, [agent_states]


class DRLAlgorithm(object):
    """Base class for all DRL algorithms"""

    def fit(self, *args, **kwargs):
        """Implements the training procedure of the algorithm"""
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        self.fit(*args, **kwargs)


class REINFORCE(DRLAlgorithm):
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

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


class GuidedRewardLearningIRL(DRLAlgorithm):
    """
    Implementation of:
    “Guided Cost Learning : Deep Inverse Optimal Control via Policy Optimization,” vol. 48, 2016.
    """

    def __init__(self, model, optimizer, demo_gen_data):
        self.model = model
        self.optimizer = optimizer
        self.demo_gen_data = demo_gen_data

    def fit(self, states, actions):
        """
        Train the reward function / model using the GRL algorithm.

        Arguments:
        -----------
        :param states: list
            The states sampled using the agent / background distribution.
        :param actions: list
            The actions produced by the background distribution.
        """
        assert len(states) == len(actions)
