import pygame
import numpy as np
from pygame.math import Vector2
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


class Snake:
    def __init__(self, cell_number):
        self.reset(cell_number)
        self.new_block = False
        self.model = None
        self.graphics_loaded = False
        self.direction_to_vector = {
            "heading_up": {"left": Vector2(-1, 0), "right": Vector2(1, 0)},
            "heading_down": {"left": Vector2(1, 0), "right": Vector2(-1, 0)},
            "heading_left": {"left": Vector2(0, 1), "right": Vector2(0, -1)},
            "heading_right": {"left": Vector2(0, -1), "right": Vector2(0, 1)},
        }

    def reset(self, cell_number):
        # headx, heady = np.random.random_integers(2, cell_number - 3, size=(2,))
        headx, heady = 4, 4
        self.body = [Vector2(headx - x, heady) for x in range(3)]
        self.direction = Vector2(1, 0)

    def load_model(self, env):
        self.model = DQN.load("dqn_snake", env=env)

    def learn(self, env):
        self.model = DQN(
            "MultiInputPolicy",
            env,
            learning_rate=0.0001,
            buffer_size=1000000,
            learning_starts=50000,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=10000,
            exploration_fraction=0.5,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            max_grad_norm=10,
            tensorboard_log="./dqn_snake_tensorboard/",
        )
        self.model.learn(
            total_timesteps=int(2e5),
            log_interval=10,
            tb_log_name="rew",
            progress_bar=True,
        )
        self.model.save("dqn_snake")

    def act(self, obs):
        action, _states = self.model.predict(obs, deterministic=True)
        return action

    def evaluate(self, env, episodes):
        return evaluate_policy(self.model, env, render=True, n_eval_episodes=episodes)

    def get_head(self):
        return self.body[0]

    def my_front(self):
        return self.get_head() + self.direction

    def go_left(self):
        """ returns (direction, head pos after direction)"""
        vector = None
        if self.heading_up():
            vector = self.direction_to_vector["heading_up"]["left"]
        elif self.heading_down():
            vector = self.direction_to_vector["heading_down"]["left"]
        elif self.heading_left():
            vector = self.direction_to_vector["heading_left"]["left"]
        else:
            vector = self.direction_to_vector["heading_right"]["left"]
        return (vector, self.get_head() + vector)

    def go_right(self):
        """ returns (direction, head pos after direction)"""
        vector = None
        if self.heading_up():
            vector = self.direction_to_vector["heading_up"]["right"]
        elif self.heading_down():
            vector = self.direction_to_vector["heading_down"]["right"]
        elif self.heading_left():
            vector = self.direction_to_vector["heading_left"]["right"]
        else:
            vector = self.direction_to_vector["heading_right"]["right"]
        return (vector, self.get_head() + vector)

    def heading_up(self):
        return self.direction.y == -1

    def heading_down(self):
        return self.direction.y == 1

    def heading_left(self):
        return self.direction.x == -1

    def heading_right(self):
        return self.direction.x == 1

    def obstacle_front(self):
        return self.my_front() in self.body[1:]

    def obstacle_left(self):
        return self.go_left()[1] in self.body[1:]

    def obstacle_right(self):
        return self.go_right()[1] in self.body[1:]

    def on_edge_front(self, num_cells):
        return not (
            0 <= self.my_front().x < num_cells and 0 <= self.my_front().y < num_cells
        )

    def on_edge_left(self, num_cells):
        return not (
            0 <= self.go_left()[1].x < num_cells
            and 0 <= self.go_left()[1].y < num_cells
        )

    def on_edge_right(self, num_cells):
        return not (
            0 <= self.go_right()[1].x < num_cells
            and 0 <= self.go_right()[1].y < num_cells
        )

    def fruit_front(self, fruit):
        return (
            (self.heading_up() and self.get_head().y > fruit.y)
            or (self.heading_down() and self.get_head().y < fruit.y)
            or (self.heading_left() and self.get_head().x > fruit.x)
            or (self.heading_right() and self.get_head().x < fruit.x)
        )

    def fruit_behind(self, fruit):
        return (
            (self.heading_up() and self.get_head().y < fruit.y)
            or (self.heading_down() and self.get_head().y > fruit.y)
            or (self.heading_left() and self.get_head().x < fruit.x)
            or (self.heading_right() and self.get_head().x > fruit.x)
        )

    def fruit_left(self, fruit):
        return (
            (self.heading_up() and self.get_head().x > fruit.x)
            or (self.heading_down() and self.get_head().x < fruit.x)
            or (self.heading_left() and self.get_head().y < fruit.y)
            or (self.heading_right() and self.get_head().y > fruit.y)
        )

    def fruit_right(self, fruit):
        return (
            (self.heading_up() and self.get_head().x < fruit.x)
            or (self.heading_down() and self.get_head().x > fruit.x)
            or (self.heading_left() and self.get_head().y > fruit.y)
            or (self.heading_right() and self.get_head().y < fruit.y)
        )

    def len(self):
        return len(self.body) - 3

    def add_block(self):
        self.new_block = True

    def check_if_crashed(self, cell_number):
        if not (0 <= self.body[0].x < cell_number) or not (
            0 <= self.body[0].y < cell_number
        ):
            return True
        if self.get_head() in self.body[1:]:
            return True
        return False

    def render(
        self,
        screen,
        cell_size,
    ):
        self.load_graphics()
        self.update_head_graphics()
        self.update_tail_graphics()

        for index, block in enumerate(self.body):
            x_pos = int(block.x * cell_size)
            y_pos = int(block.y * cell_size)
            block_rect = pygame.Rect(x_pos, y_pos, cell_size, cell_size)

            if index == 0:
                screen.blit(self.head, block_rect)
            elif index == len(self.body) - 1:
                screen.blit(self.tail, block_rect)
            else:
                previous_block = self.body[index + 1] - block
                next_block = self.body[index - 1] - block
                if previous_block.x == next_block.x:
                    screen.blit(self.body_vertical, block_rect)
                elif previous_block.y == next_block.y:
                    screen.blit(self.body_horizontal, block_rect)
                else:
                    if (
                        previous_block.x == -1
                        and next_block.y == -1
                        or previous_block.y == -1
                        and next_block.x == -1
                    ):
                        screen.blit(self.body_tl, block_rect)
                    elif (
                        previous_block.x == -1
                        and next_block.y == 1
                        or previous_block.y == 1
                        and next_block.x == -1
                    ):
                        screen.blit(self.body_bl, block_rect)
                    elif (
                        previous_block.x == 1
                        and next_block.y == -1
                        or previous_block.y == -1
                        and next_block.x == 1
                    ):
                        screen.blit(self.body_tr, block_rect)
                    elif (
                        previous_block.x == 1
                        and next_block.y == 1
                        or previous_block.y == 1
                        and next_block.x == 1
                    ):
                        screen.blit(self.body_br, block_rect)

    def update_head_graphics(self):
        head_relation = self.body[1] - self.body[0]
        if head_relation == Vector2(1, 0):
            self.head = self.head_left
        elif head_relation == Vector2(-1, 0):
            self.head = self.head_right
        elif head_relation == Vector2(0, 1):
            self.head = self.head_up
        elif head_relation == Vector2(0, -1):
            self.head = self.head_down

    def update_tail_graphics(self):
        tail_relation = self.body[-2] - self.body[-1]
        if tail_relation == Vector2(1, 0):
            self.tail = self.tail_left
        elif tail_relation == Vector2(-1, 0):
            self.tail = self.tail_right
        elif tail_relation == Vector2(0, 1):
            self.tail = self.tail_up
        elif tail_relation == Vector2(0, -1):
            self.tail = self.tail_down

    def load_graphics(self):
        if self.graphics_loaded:
            return
        self.graphics_loaded = True
        self.head_up = pygame.image.load("Graphics/head_up.png").convert_alpha()
        self.head_down = pygame.image.load("Graphics/head_down.png").convert_alpha()
        self.head_right = pygame.image.load("Graphics/head_right.png").convert_alpha()
        self.head_left = pygame.image.load("Graphics/head_left.png").convert_alpha()

        self.tail_up = pygame.image.load("Graphics/tail_up.png").convert_alpha()
        self.tail_down = pygame.image.load("Graphics/tail_down.png").convert_alpha()
        self.tail_right = pygame.image.load("Graphics/tail_right.png").convert_alpha()
        self.tail_left = pygame.image.load("Graphics/tail_left.png").convert_alpha()

        self.body_vertical = pygame.image.load(
            "Graphics/body_vertical.png"
        ).convert_alpha()
        self.body_horizontal = pygame.image.load(
            "Graphics/body_horizontal.png"
        ).convert_alpha()

        self.body_tr = pygame.image.load("Graphics/body_tr.png").convert_alpha()
        self.body_tl = pygame.image.load("Graphics/body_tl.png").convert_alpha()
        self.body_br = pygame.image.load("Graphics/body_br.png").convert_alpha()
        self.body_bl = pygame.image.load("Graphics/body_bl.png").convert_alpha()
