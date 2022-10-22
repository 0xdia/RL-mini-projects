import gym
from gym import spaces
import pygame
import numpy as np


class GridWorldEnv(gym.Env):
    metadata = {"render_fps": 4}

    def __init__(self, size=5):
        self.size = size
        self.window_size = 512
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target_direction": spaces.Box(0, 1, shape=(4,), dtype=int),
                "edge_around": spaces.Box(0, 1, shape=(4,), dtype=int)
            }
        )
        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }
        self.direction = None
        self.window = None
        self.clock = None
        self.rendering = False

    def _get_obs(self):
        return {
            "agent": self._agent_location,
            "target": self._target_location,
            "target_direction": np.array(
                [
                    int(self._agent_location[1] > self._target_location[1]),
                    int(self._agent_location[1] < self._target_location[1]),
                    int(self._agent_location[0] > self._target_location[0]),
                    int(self._agent_location[0] < self._target_location[0]),
                ]
            ),
            "edge_around": np.array(
                [
                    int(self._agent_location[1] == 0), # up
                    int(self._agent_location[1] == self.size-1), # down
                    int(self._agent_location[0] == 0), # left 
                    int(self._agent_location[0] == self.size-1), # right
                ]
            ),
        }

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def target_under_agent(self):
        for part in self.agent_body:
            if (
                part[0] == self._target_location[0]
                and part[1] == self._target_location[1]
            ):
                return True
        return False

    def distance_to_target(self):
        return abs(self._agent_location[0] - self._target_location[0]) + abs(
            self._agent_location[1] - self._target_location[1]
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._agent_location = self.np_random.integers(
            0, self.size, size=2, dtype=int
        )  # head
        self.agent_body = [(self._agent_location[0], self._agent_location[1])]
        self._target_location = self._agent_location
        while self.target_under_agent():
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )
        return self._get_obs()

    def direct(self, action):
        if self.direction is None:
            return self._action_to_direction[action]
        if self.direction[0] == 0:
            if action == 0 or action == 2:
                return self._action_to_direction[action]
        if self.direction[1] == 0:
            if action == 1 or action == 3:
                return self._action_to_direction[action]
        return self.direction

    def step(self, action):
        reward, done = 0.0, False
        self.direction = self.direct(action)
        self._agent_location = self._agent_location + self.direction
        new_body = [self._agent_location]
        new_body.extend(self.agent_body)
        self.agent_body = new_body
        if not (
            0 <= self._agent_location[0] < self.size
            and 0 <= self._agent_location[1] < self.size
        ):
            reward, done = -5, True
        if np.array_equal(self._agent_location, self._target_location):
            reward = 10.0
            while self.target_under_agent():
                self._target_location = self.np_random.integers(
                    0, self.size, size=2, dtype=int
                )
        else:
            reward = -0.1
            self.agent_body.pop()
        if self.crashed():
            reward, done = -5, True
        return self._get_obs(), reward, done, self._get_info()

    def crashed(self):
        for part in self.agent_body[1:]:
            if part[0] == self.agent_body[0][0] and part[1] == self.agent_body[0][1]:
                return True
        return False

    def render(self, mode=None):
        self._init_render()
        self._render_frame()

    def _render_frame(self):
        self.canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size
        pygame.draw.rect(
            self.canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )

        pygame.draw.circle(
            self.canvas,
            (0, 255, 0),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        for part in self.agent_body[1:]:
            pygame.draw.circle(
                self.canvas,
                (0, 0, 255),
                (np.array([part[0], part[1]]) + 0.5) * pix_square_size,
                pix_square_size / 3,
            )
        for x in range(self.size + 1):
            pygame.draw.line(
                self.canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                self.canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        self.window.blit(self.canvas, self.canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def _init_render(self):
        if self.rendering:
            return
        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode((self.window_size, self.window_size))
        self.clock = pygame.time.Clock()
        self.canvas = pygame.Surface((self.window_size, self.window_size))
        self.rendering = True

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
