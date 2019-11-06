# coding: utf-8

import tensorflow as tf
from xikarl.algorithms.base import DefaultAlgorithm, MLP

tk = tf.keras


class DDPG(DefaultAlgorithm):

    def __init__(
            self,
            actor_units=(32, 32),
            actor_scale=1,
            **kwargs):
        if "name" not in kwargs.keys():
            kwargs["name"] = "DDPG"
        super().__init__(**kwargs)

        self.actor = 1
