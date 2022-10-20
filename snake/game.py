import numpy as np
import gym
from gym import spaces
from pygame.math import Vector2

import fruit

CELL_SIZE = 35

NUM_OF_FRUITS = 0
EATING_REWARD = 0


class GameGymEnv(gym.Env):
    def __init__(self, snakes, num_of_cells):
        super(GameGymEnv, self).__init__()
        self.num_of_cells = num_of_cells
        self.snakes = snakes
        self.fruits = []

        self.observation_space = spaces.Box(
            -1, 0, shape=(1, 3), dtype=np.int8
        )
        self.action_space = spaces.Discrete(4)

    def reset(self):
        super().reset()
        self.snakes.reset(self.num_of_cells)
        # self.add_fruits()
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
        """
        state = [[0] * self.num_of_cells for _ in range(self.num_of_cells)]
        for y, x in self.snakes.body:
            if (0 <= x < self.num_of_cells) and (0 <= y < self.num_of_cells):
                state[int(x)][int(y)] = 1
        for _ in range(NUM_OF_FRUITS - len(self.fruits)):
            f = fruit.Fruit(self.num_of_cells)
            if state[f.x][f.y] == 0:
                self.fruits.append(f)
        for f in self.fruits:
            state[f.x][f.y] = 5"""
        state = [0, 0, 0]
        if self.snakes.direction.x == 1:  # snake heading right
            if (
                self.snakes.get_head() + Vector2(1, 0) in self.snakes.body[1:]
                or self.snakes.get_head().x + 1 == self.num_of_cells
            ):  # check front
                state[0] = -1
            if (
                self.snakes.get_head() + Vector2(0, -1) in self.snakes.body[1:]
                or self.snakes.get_head().y - 1 < 0
            ):  # check left
                state[1] = -1
            if (
                self.snakes.get_head() + Vector2(0, 1) in self.snakes.body[1:]
                or self.snakes.get_head().y + 1 == self.num_of_cells
            ):  # check right
                state[2] = -1

        elif self.snakes.direction.x == -1:  # snake heading left
            if (
                self.snakes.get_head() + Vector2(-1, 0) in self.snakes.body[1:]
                or self.snakes.get_head().x - 1 < 0
            ):  # check front
                state[0] = -1
            if (
                self.snakes.get_head() + Vector2(0, -1) in self.snakes.body[1:]
                or self.snakes.get_head().y - 1 < 0
            ):  # check right
                state[2] = -1
            if (
                self.snakes.get_head() + Vector2(0, 1) in self.snakes.body[1:]
                or self.snakes.get_head().y + 1 == self.num_of_cells
            ):  # check left
                state[1] = -1

        if self.snakes.direction.y == 1:  # snake heading down
            if (
                self.snakes.get_head() + Vector2(0, 1) in self.snakes.body[1:]
                or self.snakes.get_head().y + 1 == self.num_of_cells
            ):  # check front
                state[0] = -1
            if (
                self.snakes.get_head() + Vector2(-1, 0) in self.snakes.body[1:]
                or self.snakes.get_head().x - 1 < 0
            ):  # check right
                state[2] = -1
            if (
                self.snakes.get_head() + Vector2(1, 0) in self.snakes.body[1:]
                or self.snakes.get_head().x + 1 == self.num_of_cells
            ):  # check left
                state[1] = -1

        elif self.snakes.direction.y == -1:  # snake heading up
            if (
                self.snakes.get_head() + Vector2(0, -1) in self.snakes.body[1:]
                or self.snakes.get_head().y - 1 < 0
            ):  # check front
                state[0] = -1
            if (
                self.snakes.get_head() + Vector2(1, 0) in self.snakes.body[1:]
                or self.snakes.get_head().x + 1 == self.num_of_cells
            ):  # check right
                state[2] = -1
            if (
                self.snakes.get_head() + Vector2(-1, 0) in self.snakes.body[1:]
                or self.snakes.get_head().x - 1 < 0
            ):  # check left
                state[1] = -1

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
        # if self.check_eats():
        # rewards = EATING_REWARD
        # self.add_fruits()
        if self.check_collision():
            dones = True
            rewards = -2
        else:
            rewards = 1
            # rewards = -self.distance()
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
