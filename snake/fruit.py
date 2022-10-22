from pygame.math import Vector2
import numpy as np
import pygame


class Fruit:
    def __init__(self, number_of_cells):
        self.x = self.y = self.pos = None
        self.randomize(number_of_cells)

    def randomize(
        self,
        number_of_cells,
    ):
        self.x, self.y = np.random.random_integers(0, number_of_cells - 1, size=(2,))
        self.pos = Vector2(self.x, self.y)

    def draw_fruit(self, screen, cell_size, apple):
        fruit_rect = pygame.Rect(
            int(self.pos.x * cell_size),
            int(self.pos.y * cell_size),
            cell_size,
            cell_size,
        )
        screen.blit(apple, fruit_rect)
