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
    def __call__(self, states, agent_states=None):
        """
        Selects agent actions.

        :param states: state of the environment
        :param agent_states: hidden states (in the case of RNNs)
        :return: action and agent states
        """
        if agent_states is None:
            agent_states = [None] * len(states)
        state, agent_states = states[0][-1], agent_states[0]
        state, _ = self.states_preprocessor(state, self.action_selector.actions)
        state = torch.from_numpy(state).long().to(self.device)
        agent_states = list(agent_states)
        x = [state] + agent_states
        outputs = self.model(x)
        if isinstance(outputs, list):  # RNN case
            probs_v = outputs[0]
            agent_states = outputs[1:]
            if isinstance(agent_states[0], tuple) and len(agent_states[0]) == 2:  # LSTM case
                hidden, cell = agent_states[0]
            else:  # in GRU cell is None
                hidden, cell = agent_states[0], None
            agent_states = [hidden, cell] + agent_states[1:]
        else:  # trans-decoder
            probs_v = outputs
        if self.apply_softmax:
            probs_v = torch.softmax(probs_v, dim=-1)
        probs = probs_v.data.cpu().squeeze().numpy()
        action = self.action_selector(probs)
        return action, [agent_states]
