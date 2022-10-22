import numpy as np
import gym
from gym import spaces
import pygame
from pygame.math import Vector2
from time import sleep

import fruit

CELL_SIZE = 35

NUM_OF_FRUITS = 1
EATING_REWARD = 200.0


class GameGymEnv(gym.Env):
    metadata = {"render_fps": 200}

    def __init__(self, snakes, num_of_cells):
        super(GameGymEnv, self).__init__()
        self.num_of_cells = num_of_cells
        self.snakes = snakes
        self.fruits = []

        self.observation_space = spaces.Box(
            -1, num_of_cells, shape=(1, 5), dtype=np.int8
        )
        self.action_space = spaces.Discrete(3)
        self.graphics_initiated = False

    def reset(self):
        super().reset()
        self.snakes.reset(self.num_of_cells)
        self.add_fruits()
        return np.array(self._get_obs())

    def add_fruits(self):
        while True:
            for j in range(len(self.fruits)):
                if self.fruits[j].pos in self.snakes.body:
                    self.fruits.pop(j)
            if len(self.fruits) == NUM_OF_FRUITS:
                break
            current_num_fruits = len(self.fruits)
            self.fruits.extend(
                [
                    fruit.Fruit(self.num_of_cells)
                    for _ in range(NUM_OF_FRUITS - current_num_fruits)
                ]
            )

    def direct(self, action):
        if action == 0:
            pass  # heading front, direction not changed
        else:  # action == 1 is left
            self.snakes.direction = (
                self.snakes.go_left()[0] if action == 1 else self.snakes.go_right()[0]
            )
        self.snakes.body.insert(0, self.snakes.body[0] + self.snakes.direction)
        if self.snakes.new_block == True:
            self.snakes.new_block = False
        else:
            self.snakes.body.pop(-1)

    def _get_obs(self):
        state = [
            -(
                self.snakes.obstacle_front()
                or self.snakes.on_edge_front(self.num_of_cells)
            ),
            -(
                self.snakes.obstacle_left()
                or self.snakes.on_edge_left(self.num_of_cells)
            ),
            -(
                self.snakes.obstacle_right()
                or self.snakes.on_edge_right(self.num_of_cells)
            ),
            self.fruits[0].x,
            self.fruits[0].y,
        ]
        return np.array(state)

    def _get_info(self):
        return {"snakes_heads": self.snakes.body[0], "fruits": len(self.fruits)}

    def distance(self):
        return np.abs(self.snakes.body[0].x - self.fruits[0].x) + np.abs(
            self.snakes.body[0].y - self.fruits[0].y
        )

    def step(self, action):
        dones = False
        rewards = 0
        self.direct(action)
        if self.check_eats():
            rewards = EATING_REWARD
            self.add_fruits()
        if self.check_collision():
            dones = True
            rewards = -50.0
        else:
            rewards = 0.0
        return self._get_obs(), rewards, dones, self._get_info()

    def check_eats(self):
        for i in range(len(self.fruits)):
            if self.fruits[i].pos == self.snakes.body[0]:
                self.fruits.pop(i)
                self.snakes.add_block()
                return True
        return False

    def check_collision(self):
        return self.snakes.check_if_crashed([self.snakes], self.num_of_cells)

    def render(self, mode=None):
        if not self.graphics_initiated:
            self._init_graphics()
            self.graphics_initiated = True
        self._render_frame()
        sleep(0.4)

    def _render_frame(self):
        self.screen.fill((175, 215, 70))
        self.render_grass()
        for fruit in self.fruits:
            fruit.draw_fruit(self.screen, CELL_SIZE, self.apple)
        self.snakes.render(self.screen, CELL_SIZE)
        self.render_score()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def _init_graphics(self):
        self.screen = pygame.display.set_mode(
            (self.num_of_cells * CELL_SIZE, self.num_of_cells * CELL_SIZE)
        )
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.init()
        self.apple = pygame.image.load("Graphics/apple.png").convert_alpha()
        self.game_font = pygame.font.Font("Font/PoetsenOne-Regular.ttf", 25)
        self.clock = pygame.time.Clock()

    def render_grass(self):
        grass_color = (167, 209, 61)
        for row in range(self.num_of_cells):
            if row % 2 == 0:
                for col in range(self.num_of_cells):
                    if col % 2 == 0:
                        grass_rect = pygame.Rect(
                            col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE
                        )
                        pygame.draw.rect(self.screen, grass_color, grass_rect)
            else:
                for col in range(self.num_of_cells):
                    if col % 2 != 0:
                        grass_rect = pygame.Rect(
                            col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE
                        )
                        pygame.draw.rect(self.screen, grass_color, grass_rect)

    def render_score(self):
        score_text = str(sum([len(snake.body) - 3 for snake in [self.snakes]]))
        score_surface = self.game_font.render(score_text, True, (56, 74, 12))
        score_x = int(CELL_SIZE * self.num_of_cells - 60)
        score_y = int(CELL_SIZE * self.num_of_cells - 40)
        score_rect = score_surface.get_rect(center=(score_x, score_y))
        apple_rect = self.apple.get_rect(midright=(score_rect.left, score_rect.centery))
        bg_rect = pygame.Rect(
            apple_rect.left,
            apple_rect.top,
            apple_rect.width + score_rect.width + 6,
            apple_rect.height,
        )

        pygame.draw.rect(self.screen, (167, 209, 61), bg_rect)
        self.screen.blit(score_surface, score_rect)
        self.screen.blit(self.apple, apple_rect)
        pygame.draw.rect(self.screen, (56, 74, 12), bg_rect, 2)

    def close(self):
        pygame.display.quit()
        pygame.quit()
