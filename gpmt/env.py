# Author: bbrighttaer
# Project: GPMT
# Date: 4/1/2020
# Time: 2:10 PM
# File: env.py

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import numpy as np
import gym
from gym.utils import seeding
from gpmt.data import GeneratorData
from gpmt.reward import RewardFunction


class MoleculeEnv(gym.Env):
    """
    A custom gym environment for generating molecules.

    Arguments:
    ----------
    :param actions: list or tuple
        Actions allowed in the environment. Thus, the unique set of SMILES characters.
    :param reward_func:
        Instance of ::class::RewardFunction. It provides the reward function for the environment.
    :param start_char:
        Character that denotes the beginning of a SMILES string.
    :param end_char:
        Character that denotes the end of a SMILES string during generation.
    :param max_len:
        The maximum number of characters that could be contained in a generated string.
    :param seed:
        Seed value for the numpy PRNG used in by the environment.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, actions, reward_func, start_char='<', end_char='>', max_len=100, seed=None):
        assert isinstance(reward_func, RewardFunction)
        self.reward_func = reward_func
        self.start_char = start_char
        self.end_char = end_char
        self.max_len = max_len
        self.action_space = MolDiscrete(n=len(actions), all_chars=actions)
        self.observation_space = MolDiscrete(n=max_len, all_chars=actions, dim=max_len)
        self._state = [self.start_char]
        self.np_random = None
        self.seed(seed)

    def reset(self):
        self._state = [self.start_char]
        return self._state

    def step(self, action):
        assert isinstance(action, str) and len(action) == 1
        assert self.action_space.contains(action), 'Selected action is out of range.'
        prev_state = copy.deepcopy(self._state)
        state = self._state + [action]
        use_mc = action != self.end_char
        reward = self.reward_func(np.array(state), use_mc)
        if len(self._state) == self.max_len or self._state[-1] == self.end_char:
            done = True
        else:
            done = False
        info = {'prev_state': prev_state}
        self._state = list(state)
        next_state = np.array(self._state, dtype=np.object) if not done else None
        return next_state, reward, done, info

    def render(self, mode='human'):
        if mode == 'human':
            log(self._state)

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    def clone(self):
        return copy.deepcopy(self)


def log(s):
    print(s)


class MolDiscrete(gym.Space):

    def __init__(self, n, all_chars, dim=1):
        assert n >= 0
        self.n = n
        self.dim = dim
        self.all_chars = np.array(all_chars, dtype=np.object)
        super(MolDiscrete, self).__init__((), np.object)

    def sample(self):
        s = self.all_chars[self.np_random.randint(0, len(self.all_chars), self.np_random.randint(1, self.dim + 1))]
        if self.dim == 1:
            s = s[0]
        return s

    def contains(self, x):
        return self.__contains__(x)

    def __repr__(self):
        return "MolDiscrete(%d)" % self.n

    def __eq__(self, other):
        return isinstance(other, MolDiscrete) and self.n == other.n

    def __contains__(self, x):
        return x in self.all_chars
