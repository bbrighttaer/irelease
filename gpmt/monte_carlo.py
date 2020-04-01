# Author: bbrighttaer
# Project: GPMT
# Date: 4/1/2020
# Time: 4:42 PM
# File: monte_carlo.py
# Credits: https://github.com/int8/monte-carlo-tree-search

from __future__ import absolute_import, division, print_function, unicode_literals

from collections import defaultdict

import numpy as np

from gpmt.reward import RewardFunction


class MoleculeMonteCarloTreeSearchNode(object):
    def __init__(self, state, reward_func, policy, all_characters, max_len=100, parent=None, action=None,
                 end_char='>'):
        assert isinstance(reward_func, RewardFunction)
        self.state = state
        self.reward_func = reward_func
        self.policy = policy
        self.parent = parent
        self.all_characters = all_characters
        self.action = action
        self.max_len = max_len
        self.children = []
        self._num_visits = 0
        self._results = defaultdict(float)
        self._untried_actions = None
        self._done = False
        self.end_char = end_char

    @property
    def untried_actions(self):
        if self._untried_actions is None:
            self._untried_actions = self.all_characters
        return self._untried_actions

    @property
    def q(self):
        values = list(self._results.values())
        avg = np.sum(values)
        return avg

    @property
    def n(self):
        return self._num_visits

    def expand(self):
        action = self.untried_actions.pop()
        next_state = np.concatenate([self.state, list(action)])
        child_node = MoleculeMonteCarloTreeSearchNode(next_state, self.reward_func, self.policy, self.all_characters,
                                                      parent=self, action=action, max_len=self.max_len,
                                                      end_char=self.end_char)
        self.children.append(child_node)
        return child_node

    def rollout(self):
        state = np.copy(self.state)
        for _ in range(self.max_len):
            action = self.policy(state)
            state = np.concatenate([self.state, list(action)])
            if action == self.end_char:
                break
        self._done = True
        reward = self.reward_func(state, use_mc=False)
        return reward

    def backpropagate(self, result, node):
        self._num_visits += 1
        self._results[node] = result
        if self.parent:
            self.parent.backpropagate(self.q, self)

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def is_terminal_node(self):
        return self._done

    def best_child(self, c_param=1.4):
        weights = [
            (c.q / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n)) for c in self.children
        ]
        return self.children[np.argmax(weights)]


class MonteCarloTreeSearch(object):

    def __init__(self, node):
        self.root = node

    def __call__(self, simulations_number):
        rewards = []
        for _ in range(simulations_number):
            v = self._tree_policy()
            reward = v.rollout()
            rewards.append(reward)
            v.backpropagate(reward)
        # node = self.root.best_child(c_param=0.)
        avg = np.mean(rewards)
        return avg

    def _tree_policy(self):
        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node
