# coding: utf-8

import numpy as np
import tensorflow as tf

tk = tf.keras


class DefaultAlgorithm:

    def __init__(self, name="DefaultAlgo"):
        self.name = name

    @staticmethod
    def get_obs_act_size(env):
        return (
            len(env.observation_space.high),
            len(env.action_space.high)
        )

    @staticmethod
    def obs_size(env):
        return len(env.observation_space.high)

    @staticmethod
    def act_size(env):
        return len(env.action_space.high)

