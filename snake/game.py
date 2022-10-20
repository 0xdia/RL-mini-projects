import numpy as np
import gym
from gym import spaces
from pygame.math import Vector2

import fruit

CELL_SIZE = 35

NUM_OF_APPLES = 1
EATING_REWARD = 1


class GameGymEnv(gym.Env):
    def __init__(self, snakes, num_of_cells):
        super(GameGymEnv, self).__init__()
        self.num_of_cells = num_of_cells
        self.snakes = snakes
        self.fruits = []

        self.observation_space = spaces.Box(
            0, 5, shape=(self.num_of_cells, self.num_of_cells), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(4)

    def reset(self):
        super().reset()
        self.snakes.reset(self.num_of_cells)
        for part in self.snakes.body:
            for j in range(len(self.fruits)):
                if part == self.fruits[j].pos:
                    self.fruits.pop(j)
                    break
        return np.array(self._get_obs())

    def direct(self, action):
        if action == 0:
            if self.snakes.direction.y != 1:
                self.snakes.direction = Vector2(0, -1)  # up
        elif action == 1:
            if self.snakes.direction.x != -1:
                self.snakes.direction = Vector2(1, 0)  # right
        elif action == 2:
            if self.snakes.direction.y != -1:
                self.snakes.direction = Vector2(0, 1)  # down
        else:
            if self.snakes.direction.x != 1:
                self.snakes.direction = Vector2(-1, 0)  # left
        self.snakes.body.insert(0, self.snakes.body[0] + self.snakes.direction)
        if self.snakes.new_block == True:
            self.snakes.new_block = False
        else:
            self.snakes.body.pop(-1)

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

    def _get_info(self):
        return {"snakes_heads": self.snakes.body[0], "fruits": len(self.fruits)}

    def distance(self):
        # check the presence of fruits
        if len(self.fruits) > 0:
            return np.abs(self.snakes.body[0].x - self.fruits[0].x) + np.abs(
                self.snakes.body[0].y - self.fruits[0].y
            )
        return 0

    def step(self, action):
        dones = False
        rewards = 0
        self.direct(action)
        if self.check_eats():
            rewards = EATING_REWARD
        elif self.check_collision():
            dones = True
            rewards = -50
        else:
            rewards = -self.distance()
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

    def render(self):
        self._render_frame()

    def _render_frame(self):
        pass

    def close(self):
        pass
