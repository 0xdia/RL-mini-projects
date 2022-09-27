import torch
import random
from collections import deque
import numpy as np
import heapq


E = 1e-4
B = .5

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        self.seed = random.seed(seed)
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.sample_indeces = np.empty((0, 1))

    def add(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def _sample(self):
        return random.sample(self.buffer, k=self.batch_size)

    def unzip_transitions(self, experiences):
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float()
        return (states, actions, rewards, next_states, dones)

    def sample(self):
        experiences = self._sample()
        return self.unzip_transitions(experiences)

    def __len__(self):
        return len(self.buffer)


class PriorityReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, batch_size, a, seed):
        super().__init__(buffer_size, batch_size, seed)
        self.a = a
        self.e = E
        self.buffer = np.empty((0, 7))
        self.buffer_size = buffer_size
        self.is_weights = np.empty((0, 1))
        

    def add(self, state, action, reward, next_state, done, gamma):
        buffer_list = self.buffer.tolist()
        if self.__len__() == self.buffer_size:
            heapq.heappop(buffer_list)
        max_priority = np.max(np.array(self.buffer)[:, 0]) if self.__len__() != 0 else 1.
        heapq.heappush(buffer_list, [max_priority, state.tolist(), action, reward, next_state.tolist(), done, gamma])
        self.buffer = np.array(buffer_list)

    def update_priorities(self, td_errs):
        td_errs = td_errs.detach().numpy().reshape(self.batch_size,)
        self.buffer[self.sample_indeces, 0] = (np.abs(td_errs) + self.e) ** self.a
    
    def get_is_weights(self):
        return torch.tensor(self.is_weights).float()

    def _sample(self):
        processed_errs = self.buffer[:, 0]
        exp_probas = (processed_errs / np.sum(processed_errs)).astype('float64')
        # consider using np.ix_
        self.sample_indeces = np.random.choice(self.__len__(), size=self.batch_size, replace=False, p=exp_probas)
        self.is_weights = (self.batch_size * exp_probas[self.sample_indeces]) ** -B
        self.is_weights = self.is_weights / np.max(self.is_weights)
        return self.buffer[self.sample_indeces, 1:]

    def sample(self):
        experiences = self._sample()
        gammas = torch.from_numpy(np.vstack([e[-1] for e in experiences if e is not None])).float()
        return self.unzip_transitions(experiences) + (gammas, )