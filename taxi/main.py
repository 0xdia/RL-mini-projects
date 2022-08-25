from agent import Agent
from monitor import interact
import gym
import numpy as np

env = gym.make('Taxi-v3')
agent = Agent()
avg_rewards, best_avg_reward = interact(env, agent, num_episodes=1000, save_policy=True)