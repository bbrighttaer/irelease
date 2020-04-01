# Author: bbrighttaer
# Project: GPMT
# Date: 4/1/2020
# Time: 2:31 PM
# File: reward.py

from __future__ import absolute_import, division, print_function, unicode_literals

from gpmt.monte_carlo import MoleculeMonteCarloTreeSearchNode, MonteCarloTreeSearch


class RewardFunction(object):

    def __init__(self, reward_net, policy, actions, mc_max_sims=50, max_len=100, end_char='>'):
        self.net = reward_net
        self.actions = actions
        self.policy = policy
        self.mc_max_sims = mc_max_sims
        self.max_len = max_len
        self.end_char = end_char

    def __call__(self, x, use_mc):
        if use_mc:
            mc_node = MoleculeMonteCarloTreeSearchNode(x, self, self.policy, self.actions, self.max_len,
                                                       end_char=self.end_char)
            mcts = MonteCarloTreeSearch(mc_node)
            reward = mcts(simulations_number=self.mc_max_sims)
            return reward
        else:
            # Get reward of completed string using the reward net
            return 0
