import pygame, random
from pygame.math import Vector2
import numpy as np


class Fruit:
    def __init__(self, cell_number):
        self.randomize(cell_number)

    def draw_fruit(self, screen, cell_size, apple):
        fruit_rect = pygame.Rect(
            int(self.pos.x * cell_size),
            int(self.pos.y * cell_size),
            cell_size,
            cell_size,
        )
        screen.blit(apple, fruit_rect)
        # pygame.draw.rect(screen,(126,166,114),fruit_rect)

    def randomize(
        self,
        cell_number,
    ):
        self.x, self.y = np.random.random_integers(0, cell_number - 1, size=(2,))
        self.pos = Vector2(self.x, self.y)
