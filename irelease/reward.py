# Author: bbrighttaer
# Project: GPMT
# Date: 4/1/2020
# Time: 2:31 PM
# File: reward.py

from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from irelease.monte_carlo import MoleculeMonteCarloTreeSearchNode, MonteCarloTreeSearch
from irelease.utils import canonical_smiles, seq2tensor


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
                 expert_func=None, use_mc=True, no_mc_fill_val=0.0, use_true_reward=False, true_reward_func=None):
        if use_true_reward:
            assert (true_reward_func is not None), 'If true reward should be used then the ' \
                                                   'true reward function must be supplied'
            assert (callable(true_reward_func))
        self.model = reward_net
        self.actions = actions
        self.mc_policy = mc_policy
        self.mc_max_sims = mc_max_sims
        self.max_len = max_len
        self.end_char = end_char
        self.device = device
        self.expert_func = expert_func
        self.true_reward_func = true_reward_func
        self.mc_enabled = use_mc
        self.no_mc_fill_val = no_mc_fill_val
        self.use_true_reward = use_true_reward

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
            if self.mc_enabled:
                mc_node = MoleculeMonteCarloTreeSearchNode(x, self, self.mc_policy, self.actions, self.max_len,
                                                           end_char=self.end_char)
                mcts = MonteCarloTreeSearch(mc_node)
                reward = mcts(simulations_number=self.mc_max_sims)
                return reward
            else:
                return self.no_mc_fill_val
        else:
            # Get reward of completed string using the reward net or a given reward function.
            state = ''.join(x.tolist())
            if self.use_true_reward:
                state = state[1:-1].replace('\n', '-')
                reward = self.true_reward_func(state, self.expert_func)
            else:
                smiles, valid_vec = canonical_smiles([state])
                valid_vec = torch.tensor(valid_vec).view(-1, 1).float().to(self.device)
                inp, _ = seq2tensor([state], tokens=self.actions)
                inp = torch.from_numpy(inp).long().to(self.device)
                reward = self.model([inp, valid_vec]).squeeze().item()
            return reward

    def expert_reward(self, x):
        if self.expert_func:
            return self.expert_func(x)
        return None
