# coding: utf-8

import tensorflow as tf
from cached_property import cached_property
tk = tf.keras


class DefaultAlgorithm:

    def __init__(self, env, name="DefaultAlgo"):
        self.env = env
        self.name = name

    @cached_property
    def obs_size(self):
        return len(self.env.observation_space.high)

    @cached_property
    def act_size(self):
        return len(self.env.action_space.high)

