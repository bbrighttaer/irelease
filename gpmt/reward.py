# Author: bbrighttaer
# Project: GPMT
# Date: 4/1/2020
# Time: 2:31 PM
# File: reward.py

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import torch

from gpmt.monte_carlo import MoleculeMonteCarloTreeSearchNode, MonteCarloTreeSearch
from gpmt.utils import canonical_smiles, seq2tensor


class RewardFunction:
    """
    Provides operations pertaining to the reward function in the simulation environment.

    Arguments:
    ----------
    :param reward_net: nn.Module
        Neural net that parameterizes the reward function. This is trained using IRL.
    :param mc_policy:
        The policy to be used for Monte Carlo Tree Search.
    :param actions:
        All allowed actions in the simulation environment. In the molecule case, these are the unique tokens or chars.
    :param mc_max_sims:
        Maximum number of Monte Carlo Tree Search simulations to perform
    :param max_len:
        Maximum length of a generated SMILES string.
    :param end_char:
        Character denoting the end of a SMILES string generation process.
    :param expert_func: callable
        A function that implements the true or expert's reward function to be used to monitor how well the
        parameterized reward function is doing. This callback function shall take a single argument: the state, x
    """

    def __init__(self, reward_net, mc_policy, actions, mc_max_sims=50, max_len=100, end_char='>', device='cpu',
                 expert_func=None):
        if expert_func:
            assert callable(expert_func)
        self.model = reward_net
        self.actions = actions
        self.mc_policy = mc_policy
        self.mc_max_sims = mc_max_sims
        self.max_len = max_len
        self.end_char = end_char
        self.device = device
        self.expert_func = expert_func

    @torch.no_grad()
    def __call__(self, x, use_mc):
        """
        Calculates the reward function of a given state.

        :param x:
            The state to be used in calculating the reward.
        :param use_mc:
            Whether Monte Carlo Tree Search or the parameterized reward function should be used
        :return: float
            A scalar value representing the reward w.r.t. the given state x.
        """
        if use_mc:
            mc_node = MoleculeMonteCarloTreeSearchNode(x, self, self.mc_policy, self.actions, self.max_len,
                                                       end_char=self.end_char)
            mcts = MonteCarloTreeSearch(mc_node)
            reward = mcts(simulations_number=self.mc_max_sims)
            return reward
        else:
            # Get reward of completed string using the reward net
            state = ''.join(x.tolist())
            # _, valid_vec = canonical_smiles([state])
            # valid_vec = torch.tensor(valid_vec).view(-1, 1).float().to(self.device)
            # inp, _ = seq2tensor([state], tokens=self.actions)
            # inp = torch.from_numpy(inp).long().to(self.device)
            # reward = self.model([inp, valid_vec]).squeeze().item()
            reward = self.model(state).squeeze().item()
            return reward

    def expert_reward(self, x):
        if self.expert_func:
            return self.expert_func(x)
        return None
