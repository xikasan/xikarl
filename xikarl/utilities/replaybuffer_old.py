# coding: utf-8

import random
import numpy as np
from collections import namedtuple


TimeStep = namedtuple("TimeStep", ["state", "action", "next_state", "reward", "done"])


class ReplayBuffer:

    def __init__(self, timestep=TimeStep):
        self._buf = []
        self.timestep = timestep

    def __len__(self):
        return len(self._buf)

    def flush(self):
        self._buf = []

    def buffer(self, data=None):
        if data is not None:
            self._buf = data
        return self._buf

    def push_back(self, data):
        self._buf.append(data)

    def sample(self, batch_size):
        return random.sample(self._buf, batch_size)

    def get_batch(self, size, dtype=TimeStep):
        time_steps = self.sample(size)
        time_steps = [np.array(content) for content in [*zip(*time_steps)]]
        return dtype(*time_steps)

    def get_last(self, size, timestep=None):
        if timestep is None:
            timestep = self.timestep
        time_steps = self._buf if size > len(self) else self._buf[len(self)-size:]
        return timestep(*zip(*time_steps))

    def as_tuple(self):
        if len(self) == 0:
            return []
        return self.timestep(*zip(*self._buf))

    def as_dict(self):
        if len(self) == 0:
            return {}
        return self.as_tuple()._asdict()
