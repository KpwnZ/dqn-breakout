from typing import (
    List,
    Tuple,
)

import torch
import numpy as np
import random

from utils_types import (
    BatchAction,
    BatchDone,
    BatchNext,
    BatchReward,
    BatchState,
    TensorStack5,
    TorchDevice,
)


# class ReplayMemory(object):

#     def __init__(
#             self,
#             channels: int,
#             capacity: int,
#             device: TorchDevice,
#     ) -> None:
#         self.__device = device
#         self.__capacity = capacity
#         self.__size = 0
#         self.__pos = 0

#         self.__m_states = torch.zeros(
#             (capacity, channels, 84, 84), dtype=torch.uint8)
#         self.__m_actions = torch.zeros((capacity, 1), dtype=torch.long)
#         self.__m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
#         self.__m_dones = torch.zeros((capacity, 1), dtype=torch.bool)

#     def push(
#             self,
#             folded_state: TensorStack5,
#             action: int,
#             reward: int,
#             done: bool,
#     ) -> None:
#         self.__m_states[self.__pos] = folded_state
#         self.__m_actions[self.__pos, 0] = action
#         self.__m_rewards[self.__pos, 0] = reward
#         self.__m_dones[self.__pos, 0] = done

#         self.__pos = (self.__pos + 1) % self.__capacity
#         self.__size = max(self.__size, self.__pos)

#     def sample(self, batch_size: int) -> Tuple[
#             BatchState,
#             BatchAction,
#             BatchReward,
#             BatchNext,
#             BatchDone,
#     ]:
#         indices = torch.randint(0, high=self.__size, size=(batch_size,))
#         b_state = self.__m_states[indices, :4].to(self.__device).float()
#         b_next = self.__m_states[indices, 1:].to(self.__device).float()
#         b_action = self.__m_actions[indices].to(self.__device)
#         b_reward = self.__m_rewards[indices].to(self.__device).float()
#         b_done = self.__m_dones[indices].to(self.__device).float()
#         return b_state, b_action, b_reward, b_next, b_done

#     def __len__(self) -> int:
#         return self.__size

# Sum tree for prioritized experience replay, State is a 84*84 image
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = [0] * (2 * capacity - 1)
        self.data = [None] * capacity
        self.count = 0

    def _fetch(self, idx, s):
        left = 2 * idx + 1          # left child node
        right = left + 1            # right child node

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:    # smaller then the node, go to left
            return self._fetch(left, s)
        else:                       # go to right
            return self._fetch(right, s - self.tree[left])

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.count < self.capacity:
            self.count += 1

    def update(self, idx, p):
        new_v = p - self.tree[idx]
        self.tree[idx] = p
        idx = (idx - 1) // 2

        while True:
            self.tree[idx] += new_v
            if idx == 0:
                break
            idx = (idx - 1) // 2

    def sum_priority(self):
        return self.tree[0]

    def get(self, s):
        idx = self._fetch(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

# memory using sum tree
class ReplayMemory(object):
    # with sum tree
    def __init__(self, channels, capacity, device):
        self.__capacity = capacity
        self.__device = device
        self.__tree = SumTree(capacity)
        self.__pos = 0
        self.__a = 0.6
        self.__e = 0.01
        self.__max_priority = 1.0
    
    def push(self, folded_state: TensorStack5, action, reward, done):
        data = (folded_state, action, reward, done)
        priority = (np.abs(self.__max_priority) + self.__e) ** self.__a
        self.__tree.add(priority, data)
        self.__pos = (self.__pos + 1) % self.__capacity

    def sample(self, batch_size) -> Tuple[
        BatchState,
        BatchAction,
        BatchReward,
        BatchNext,
        BatchDone,
        List[int],
    ]:
        batch_indices = []
        batch_priorities = []
        segment = self.__tree.sum_priority() / batch_size

        batch_action = torch.zeros((batch_size, 1), dtype=torch.long)
        batch_reward = torch.zeros((batch_size, 1), dtype=torch.int8)
        batch_done = torch.zeros((batch_size, 1), dtype=torch.bool)
        batch_state =  torch.zeros((batch_size, 5, 84, 84), dtype=torch.uint8)
        
        weights = [0] * 32
        idxs = [0] * 32
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.__tree.get(s)
            batch_indices.append(idx)
            batch_priorities.append(p)
            weights[i] = p
            batch_action[i, 0] = data[1]
            batch_reward[i, 0] = data[2]
            batch_done[i, 0] = data[3]
            batch_state[i] = data[0]
            idxs[i] = idx

        batch_next = batch_state[:, 1:, :, :].to(self.__device).float()
        batch_state = batch_state[:, :4, :, :].to(self.__device).float()
        batch_action = batch_action.to(self.__device)
        batch_reward = batch_reward.to(self.__device).float()
        batch_done = batch_done.to(self.__device).float()

        return batch_state, batch_action, batch_reward, batch_next, batch_done, idxs

    def update(self, idxs, errors):
        self.__max_priority = max(self.__max_priority, np.max(np.abs(errors)))
        for i, idx in enumerate(idxs):
            p = (abs(errors[i]) + self.__e) ** self.__a
            self.__tree.update(idx, p)
    
    def __len__(self):
        return (self.__tree.count)


