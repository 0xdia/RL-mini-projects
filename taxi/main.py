from agent import Agent
from monitor import interact
import gym

env = gym.make('Taxi-v3')

methods = ["sarsa", "q_learning", "expected_sarsa"]

for method in methods:
  agent = Agent(alpha=0.2, gamma=0.9)
  print(f"\n[{method}]")
  print("===> Training:")
  avg_rewards, best_avg_reward = interact(env, agent, num_episodes=20000, method=method)

  print("===> Testing:")
  avg_reward, best_avg_reward = interact(env, agent, num_episodes=2000, method=method, mode="test")
  print(f"best avg rewards: {best_avg_reward}")
