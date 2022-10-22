import game
import snake


NUMBER_OF_CELLS = 10
NUMBER_OF_SNAKES = 1

snakes = snake.Snake(NUMBER_OF_CELLS)
env = game.GameGymEnv(snakes, NUMBER_OF_CELLS)

""" try:
    snakes.learn(env)
finally:
    # print(snakes.evaluate(env, 100))
    pass

import sys

sys.exit(0) """

import pygame
from time import sleep
from random import randint

env.render()
SCREEN_UPDATE = pygame.USEREVENT
pygame.time.set_timer(SCREEN_UPDATE, 50)

for _ in range(10000):
    print("Episode: ", _ + 1)
    done = False
    state = env.reset()
    while not done:
        for event in pygame.event.get():
            if event.type == SCREEN_UPDATE:
                action = randint(0, 2)
                next_obs, reward, done, _ = env.step(action)
                state = next_obs
                env.render()
                if done:
                    break
                #sleep(0.1)
