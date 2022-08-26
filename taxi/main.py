from agent import Agent
from monitor import interact
import gym

env = gym.make('Taxi-v3')
agent = Agent()

print("===> Training:")
avg_rewards, best_avg_reward = interact(env, agent, num_episodes=100000, method="expected_sarsa")
print(f"Best avg reward: {best_avg_reward}\n")

print("===> Testing:")
avg_reward = interact(env, agent, num_episodes=1000, method="expected_sarsa", mode="test")
print(f"avg rewards: {avg_reward}")