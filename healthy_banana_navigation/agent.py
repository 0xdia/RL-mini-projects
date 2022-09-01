import torch
import numpy as np
import random

from neural_network import QNeuralNetwork
from replay_buffer import ReplayBuffer


BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 0.99
ALPHA = 1e-3
LEARNING_RATE = 5e-4
UPDATE_EVERY = 4

class Agent:
    def __init__(self, state_size, action_size, seed, load_trained_model=False, model_state_path=None, train_mode=False):
        assert load_trained_model == (model_state_path is not None)

        self.seed = random.seed(seed)
        self.state_size = state_size
        self.action_size = action_size

        self.train_mode   = train_mode

        hidden_layers_size = [20, 15, 10]
        self.qnetwork_local  = QNeuralNetwork(state_size, action_size, hidden_layers_size, seed)
        self.qnetwork_target = QNeuralNetwork(state_size, action_size, hidden_layers_size, seed)
        self.optimizer = torch.optim.Adam(self.qnetwork_local.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        self.episode_num = 0
        self.t_step = 0
        self.scores = []
        if load_trained_model:
            trained_model_state = torch.load(model_state_path)
            self.qnetwork_local.load_state_dict(trained_model_state["qnetwork_local"])
            self.qnetwork_target.load_state_dict(trained_model_state["qnetwork_target"])
            #self.memory.load(trained_model_state["raplay_buffer"])
            #self.episode_num.load_state_dict(trained_model_state["last_episode"])
            #self.scores.load_state_dict(trained_model_state["scores"])
            #self.t_step = self.episode_num

    def step(self, state, action, reward, next_state, done):
        if not self.train_mode:
            return
        self.episode_num += 1
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state, eps=.1):
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        if self.train_mode:
            self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        Q_target_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (GAMMA * Q_target_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        loss = torch.nn.functional.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update()

    def soft_update(self):
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(ALPHA*local_param.data + (1.0-ALPHA)*target_param.data)

    def save_model(self, path, scores):
        assert path is not None

        model_state = {
            "qnetwork_local": self.qnetwork_local.state_dict(),
            "qnetwork_target": self.qnetwork_target.state_dict(),
            #"replay_buffer": self.memory,
            "scores": scores,
            "last_episode": self.episode_num
        }
        torch.save(model_state, path)