import torch
import numpy as np

from policy import Policy
from neural_net import QNeuralNetwork
from replay_buffer import ReplayBuffer
from exploration_noise import OrnsteinUhlenbeckProcess
from schedule import IncreasingLinearSchedule

GAMMA = 0.99
LEARNING_RATE = 1e-2
TAU = 1e-2
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 32
MODEL_PATH = './learned_policy'

class Agent:
    def __init__(self, state_size, action_size, action_low, action_high, load_model, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.action_low, self.action_high = action_low, action_high
        self.seed = seed
        self.gamma = GAMMA

        self.actor = Policy(state_size, action_size, [16, 10, 6], seed)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.actor_target = Policy(state_size, action_size, [16, 10, 6], seed)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = QNeuralNetwork(state_size, action_size, [25, 15, 5], seed)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)
        self.critic_target = QNeuralNetwork(state_size, action_size, [25, 15, 5], seed)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.reply_buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        self.random_process = OrnsteinUhlenbeckProcess(size=action_size, std=IncreasingLinearSchedule(0.1, 1, 9))

        self.model_path = MODEL_PATH
        if load_model:
            trained_model_state = torch.load(self.model_path)
            self.actor.load_state_dict(trained_model_state["policy"])

    def reset_noise(self):
        self.random_process.reset_states()

    def soft_update(self):
        for target_param, local_param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * local_param + (1 - TAU) * target_param)
        for target_param, local_param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(TAU * local_param + (1 - TAU) * target_param)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state) + self.random_process.sample()
            action = np.clip(action.detach().numpy(), self.action_low, self.action_high)
            self.prev_action, self.prev_state = action, state
            return action

    def step(self, reward, state, done):
        self.reply_buffer.add(self.prev_state, self.prev_action, reward, state, done)
        if self.reply_buffer.__len__() >= BATCH_SIZE:
            experiences = self.reply_buffer.sample()
            self.learn(experiences)
    
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        target_actions = self.actor_target(next_states)
        target_critics = rewards + self.gamma * self.critic_target(next_states, target_actions) * (1 - dones)
        expected_critics = self.critic(states, actions)

        critic_loss = torch.nn.functional.mse_loss(expected_critics, target_critics)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update()
    
    def save_model(self):
        model_state = {"policy": self.actor.state_dict()}
        torch.save(model_state, self.model_path)