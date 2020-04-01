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
    metadata = {'render.modes': ['human']}

    def __init__(self, gen_data, reward_func, start_char='<', end_char='>', max_len=100, seed=None):
        assert isinstance(gen_data, GeneratorData)
        assert isinstance(reward_func, RewardFunction)
        self.data_gen = gen_data
        self.reward_func = reward_func
        self.start_char = start_char
        self.end_char = end_char
        self.max_len = max_len
        self.action_space = MolDiscrete(n=gen_data.n_characters, all_chars=gen_data.all_characters)
        self.observation_space = MolDiscrete(n=max_len, all_chars=gen_data.all_characters, dim=max_len)
        self._state = [self.start_char]
        self.np_random = None
        self.seed(seed)

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, s):
        self._state = s

    def reset(self):
        self._state = [self.start_char]
        return self._state

    def step(self, action):
        assert isinstance(action, str) and len(action) == 1
        assert self.action_space.contains(action), 'Selected action is out of range.'
        prev_state = self._state
        self._state.append(action)
        use_mc = self._state[-1] != self.end_char
        reward = self.reward_func(self._state, use_mc)
        if len(self._state) == self.max_len or self._state[-1] == self.end_char:
            done = True
        else:
            done = False
        info = {'prev_state': prev_state}
        return np.array(self._state, dtype=np.object), reward, done, info

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
