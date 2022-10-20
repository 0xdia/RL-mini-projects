import pygame
from pygame.math import Vector2
import numpy as np
from stable_baselines3 import DQN


class Snake:
    def __init__(self, cell_number):
        headx, heady = np.random.random_integers(2, cell_number - 3, size=(2,))
        self.body = [Vector2(headx - x, heady) for x in range(3)]
        self.direction = Vector2(0, 0)
        self.new_block = False
        self.model = None

    def get_head(self):
        return self.body[0]

    def learn(self, env):
        self.model = DQN(
            "MlpPolicy",
            env,
            learning_rate=0.0001,
            buffer_size=100000,
            learning_starts=50000,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.35,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.1,
            max_grad_norm=10,
            verbose=1,
            device="auto",
            tensorboard_log="./dqn_snake_tensorboard/",
        )
        self.model.learn(
            total_timesteps=int(5e5),
            log_interval=500,
            tb_log_name="rew=10",
            progress_bar=True,
        )
        self.model.save("dqn_snake")

    def act(self, obs):
        action, _states = self.model.predict(obs, deterministic=True)
        return action

    def add_block(self):
        self.new_block = True

    def reset(self, cell_number):
        headx, heady = np.random.random_integers(2, cell_number - 3, size=(2,))
        self.body = [Vector2(headx - x, heady) for x in range(3)]
        self.direction = Vector2(0, 0)

    def check_if_crashed(self, snakes, cell_number):
        if not (0 <= self.body[0].x < cell_number) or not (
            0 <= self.body[0].y < cell_number
        ):
            return True
        for snake in snakes:
            if self != snake and self.body[0] == snake.body[0]:
                return True
            body = snake.body if snake != self else self.body[1:]
            for part in body:
                if self.body[0] == part:
                    return True
