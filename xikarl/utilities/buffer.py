# coding: utf-8

import numpy as np


def RLDataDict(model):
    return {
        "state": model.obs_size,
        "action": model.act_size,
        "next_state": model.obs_size,
        "reward": 1,
        "done": 1
    }


class Batch:

    def __init__(self, size):
        self.size = size


class ReplayBuffer:

    def __init__(self, keys_dims, capacity=36001, dtype=np.float32):
        self._buf = {key: np.empty(shape=(capacity, dim), dtype=dtype) for key, dim in keys_dims.items()}
        self.index = 0
        self.capacity = capacity
        self._is_full = False

    def __len__(self):
        return self.capacity if self._is_full else self.index+1

    def add(self, **kwargs):
        for key, val in kwargs.items():
            self._buf[key][self.index, :] = val

        if (self.index+1) == self.capacity:
            self._is_full = True
            self.index = 0
            return

        self.index += 1

    def sample(self, size):
        high = self.capacity if self._is_full else self.index
        idx = np.random.randint(0, high, size)
        return {
            key: vals[idx, :] for key, vals in self._buf.items()
        }

    def batch(self, size):
        batch_data = self.sample(size)
        batch = Batch(size)
        for key, val in batch_data.items():
            batch.__setattr__(key, val)
        return batch

    def buffer(self):
        if self._is_full:
            return self._buf
        return {
            key: vals[0:self.index, :] for key, vals in self._buf.items()
        }
