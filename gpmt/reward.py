# Author: bbrighttaer
# Project: GPMT
# Date: 4/1/2020
# Time: 2:31 PM
# File: reward.py

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from gpmt.monte_carlo import MoleculeMonteCarloTreeSearchNode, MonteCarloTreeSearch


class RewardFunction:
    """
    Provides operations pertaining to the reward function in the simulation environment.

    Arguments:
    ----------
    :param reward_net: nn.Module
        Neural net that parameterizes the reward function. This is trained using IRL.
    :param policy:
        The policy to be used for Monte Carlo Tree Search.
    :param actions:
        All allowed actions in the simulation environment. In the molecule case, these are the unique tokens or chars.
    :param mc_max_sims:
        Maximum number of Monte Carlo Tree Search simulations to perform
    :param max_len:
        Maximum length of a generated SMILES string.
    :param end_char:
        Character denoting the end of a SMILES string generation process.
    """

    def __init__(self, reward_net, policy, actions, mc_max_sims=50, max_len=100, end_char='>'):
        self.net = reward_net
        self.actions = actions
        self.policy = policy
        self.mc_max_sims = mc_max_sims
        self.max_len = max_len
        self.end_char = end_char

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
            mc_node = MoleculeMonteCarloTreeSearchNode(x, self, self.policy, self.actions, self.max_len,
                                                       end_char=self.end_char)
            mcts = MonteCarloTreeSearch(mc_node)
            reward = mcts(simulations_number=self.mc_max_sims)
            return reward
        else:
            # Get reward of completed string using the reward net
            return np.random.random()
