import numpy as np
import gym
import pygame
from gym import spaces
from pygame.math import Vector2

import fruit

CELL_SIZE = 40

NUM_OF_APPLES = 1
EATING_REWARD = 1


class GameGymEnv(gym.Env):
    metadata = {"render_fps": 65}

    def __init__(self, snakes, num_of_cells):
        super(GameGymEnv, self).__init__()
        self.num_of_cells = num_of_cells
        #self.screen = pygame.display.set_mode(
        #    (self.num_of_cells * CELL_SIZE, self.num_of_cells * CELL_SIZE)
        #)
        #pygame.mixer.pre_init(44100, -16, 2, 512)
        #pygame.init()
        #self.apple = pygame.image.load("Graphics/apple.png").convert_alpha()
        #self.game_font = pygame.font.Font("Font/PoetsenOne-Regular.ttf", 25)
        self.snakes = snakes
        self.fruits = []

        self.observation_space = spaces.Box(
            0, 5, shape=(self.num_of_cells, self.num_of_cells), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(4)

        self.clock = pygame.time.Clock()

    def reset(self):
        super().reset()
        self.snakes.reset(self.num_of_cells)
        for part in self.snakes.body:
            for j in range(len(self.fruits)):
                if part == self.fruits[j].pos:
                    self.fruits.pop(j)
                    break
        # self._render_frame()
        return np.array(self._get_obs())

    def direct(self, snake, action):
        if action == 0:
            if snake.direction.y != 1:
                snake.direction = Vector2(0, -1)  # up
        elif action == 1:
            if snake.direction.x != -1:
                snake.direction = Vector2(1, 0)  # right
        elif action == 2:
            if snake.direction.y != -1:
                snake.direction = Vector2(0, 1)  # down
        else:
            if snake.direction.x != 1:
                snake.direction = Vector2(-1, 0)  # left

    def _get_obs(self):
        state = [[0] * self.num_of_cells for _ in range(self.num_of_cells)]
        for y, x in self.snakes.body:
            if (0 <= x < self.num_of_cells) and (0 <= y < self.num_of_cells):
                state[int(x)][int(y)] = 1
        for _ in range(NUM_OF_APPLES - len(self.fruits)):
            f = fruit.Fruit(self.num_of_cells)
            if state[f.x][f.y] == 0:
                self.fruits.append(f)
        for f in self.fruits:
            state[f.x][f.y] = 5
        return state

    def get_snakes(self):
        return self._get_info()

    def _get_info(self):
        return {"snakes_heads": self.snakes.body[0], "fruits": len(self.fruits)}

    def step(self, action):
        dones = False
        rewards = 0
        self.direct(self.snakes, action)
        self.snakes.body.insert(0, self.snakes.body[0] + self.snakes.direction)
        if self.snakes.new_block == True:
            self.snakes.new_block = False
        else:
            self.snakes.body.pop(-1)
        rewards = self.check_eats(0)
        if self.check_collision():
            dones = True
            rewards = -10
        # self._render_frame()
        return self._get_obs(), rewards, dones, self._get_info()

    def check_eats(self, s):
        for i in range(len(self.fruits)):
            if self.fruits[i].pos == self.snakes.body[0]:
                self.fruits.pop(i)
                self.snakes.add_block()
                return EATING_REWARD
        return 0

    def check_collision(self):
        return self.snakes.check_if_crashed([self.snakes], self.num_of_cells)

    def render(self):
        self._render_frame()

    def _render_frame(self):
        self.screen.fill((175, 215, 70))

        self.render_grass()
        for fruit in self.fruits:
            fruit.draw_fruit(self.screen, CELL_SIZE, self.apple)
        self.snakes.render(self.screen, CELL_SIZE)
        self.render_score()

        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

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
