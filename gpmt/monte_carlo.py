# Author: bbrighttaer
# Project: GPMT
# Date: 4/1/2020
# Time: 4:42 PM
# File: monte_carlo.py
# Credits: https://github.com/int8/monte-carlo-tree-search (awesome tutorial)

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np


class MoleculeMonteCarloTreeSearchNode(object):
    """
    Implements a MCTS algorithm for simulating the structural evolution of molecules. We use this to estimate the
    reward for intermediate states (non-terminal) in the generation process.

    Arguments:
    ----------
    :param state:
        The state for the simulation.
    :param reward_func:
        Reward function for estimating the reward of a terminal node.
    :param policy:
        The rollout policy of the MCTS.
    :param all_characters:
        All actions/characters allowed in the simulation environment.
    :param max_len:
        Maximum length of a generated SMILES string.
    :param parent:
        The parent node of `state`
    :param end_char:
        Character denoting the end of a SMILES string generation process.
    """

    def __init__(self, state, reward_func, policy, all_characters, max_len=100, parent=None, end_char='>'):
        self.state = state
        self.reward_func = reward_func
        self.policy = policy
        self.parent = parent
        self.all_characters = all_characters
        self.max_len = max_len
        self.children = []
        self._num_visits = 0
        self._value = 0.
        self._untried_actions = None
        self._done = False
        self.end_char = end_char

    @property
    def untried_actions(self):
        if self._untried_actions is None:
            self._untried_actions = list(self.all_characters)
        return self._untried_actions

    @property
    def q(self):
        return self._value

    @property
    def n(self):
        return self._num_visits

    def expand(self):
        action = self.untried_actions.pop()
        next_state = np.concatenate([self.state, list(action)])
        child_node = MoleculeMonteCarloTreeSearchNode(next_state, self.reward_func, self.policy, self.all_characters,
                                                      parent=self, max_len=self.max_len, end_char=self.end_char)
        self.children.append(child_node)
        return child_node

    def rollout(self):
        state = np.copy(self.state)
        hidden_states = None
        for _ in range(self.max_len - len(state)):
            # policy must return a tuple. See ::class::PolicyAgent
            action, hidden_states = self.policy([state], hidden_states, monte_carlo=True)
            state = np.concatenate([state, list(action)])
            if action == self.end_char:
                break
        self._done = True
        reward = self.reward_func(state, use_mc=False)
        return reward

    def backpropagate(self, result):
        self._num_visits += 1
        self._value += result
        if self.parent:
            self.parent.backpropagate(self.q)

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
    """
    Performs molecule MCTS

    Argument:
    ----------
    :param node: ::class::MoleculeMonteCarloTreeSearchNode
        The node to use as the root for the MCTS.
    """

    def __init__(self, node):
        self.root = node

    def __call__(self, simulations_number):
        """Returns the average results after N molecule MCTS simulations."""
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
