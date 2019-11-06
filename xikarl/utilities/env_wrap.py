# coding: utf-8

import gym
from cached_property import cached_property


class EnvWrap(gym.Env):

    def __init__(self, env):
        self._env = env
        self.name = env.name

    @cached_property
    def action_size(self):
        return len(self._env.action_space.high)

    @cached_property
    def observation_size(self):
        return len(self._env.observation_space.low)
