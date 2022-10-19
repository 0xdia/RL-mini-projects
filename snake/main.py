import pygame
import game
import gym
import snake
import random

NUMBER_OF_CELLS = 12
NUMBER_OF_SNAKES = 1

rate = 140

snakes = snake.Snake(NUMBER_OF_CELLS)
env = game.GameGymEnv(snakes, NUMBER_OF_CELLS)

SCREEN_UPDATE = pygame.USEREVENT
pygame.time.set_timer(SCREEN_UPDATE, rate)

snakes.learn(env)

print("[*] Learning phase: done")

for _ in range(100):
    print("Episode: ", _+1)
    done = False
    state = env.reset()
    while not done:
        for event in pygame.event.get():
        #    print(event)
            if event.type == SCREEN_UPDATE:
                #action = random.randint(0, 3)
                action = snakes.act(state)
                next_obs, reward, done, _ = env.step(action)
                state = next_obs
                # env.render()
                # pygame.time.set_timer(SCREEN_UPDATE, rate)
