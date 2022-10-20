import game
import gym
import snake
import random

NUMBER_OF_CELLS = 10
NUMBER_OF_SNAKES = 1

rate = 140

snakes = snake.Snake(NUMBER_OF_CELLS)
env = game.GameGymEnv(snakes, NUMBER_OF_CELLS)

snakes.learn(env)

""" for _ in range(100):
    print("Episode ", _ + 1, ":")
    score = 0
    done = False
    state = env.reset()
    while not done:
        action = snakes.act(state)
        next_obs, reward, done, _ = env.step(action)
        score += reward
        state = next_obs
"""
