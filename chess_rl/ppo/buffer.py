import numpy as np
import torch as T

from chess_rl.config import DEVICE


class RolloutBuffer:
    def __init__(self, capacity):

        self.states = []
        self.pis = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.masks = []

    def add(self, states, pis, rewards, dones, values, masks):
        self.states.append(states)
        self.pis.append(pis)
        self.rewards.append(rewards)
        self.dones.append(dones)
        self.values.append(values)
        self.masks.append(masks)

    def clear(self):
        self.states.clear()
        self.pis.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.masks.clear()

    def get(self):
        states = T.stack(self.states)
        pis = T.tensor(np.array(self.pis), dtype=T.float32, device=DEVICE)
        rewards = T.tensor(self.rewards, dtype=T.float32, device=DEVICE)
        dones = T.tensor(self.dones, dtype=T.bool, device=DEVICE)
        values = T.stack(self.values)

        masks = T.stack(self.masks).bool()

        return {
            "states": states,
            "pis": pis,
            "rewards": rewards,
            "dones": dones,
            "values": values,
            "masks": masks,
        }

    def add_last_value(self, value):
        self.values.append(value)

    def minibatch(self, buffer, batch_size):
        buffer = self.get()

        num_samples = buffer["states"].size(0)

        indices = T.randperm(num_samples, device=DEVICE)

        for start in range(0, num_samples, batch_size):
            batch_idx = indices[start : start + batch_size]

            yield (
                batch_idx,
                {
                    "states": buffer["states"][batch_idx],
                    "pis": buffer["pis"][batch_idx],
                    "rewards": buffer["rewards"][batch_idx],
                    "dones": buffer["dones"][batch_idx],
                    "values": buffer["values"][batch_idx],
                    "masks": buffer["masks"][batch_idx],
                },
            )
