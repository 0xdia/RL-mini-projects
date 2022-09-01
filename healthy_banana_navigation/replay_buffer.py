import torch
import random
from collections import namedtuple, deque
import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        self.seed = random.seed(seed)
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        self.buffer.append(self.experience(state, action, reward, next_state, done))

    def sample(self):
        experiences = random.sample(self.buffer, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.buffer)