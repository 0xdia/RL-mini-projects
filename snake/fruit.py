from pygame.math import Vector2
import numpy as np


class Fruit:
    def __init__(self, cell_number):
        self.x = self.y = self.pos = None
        self.randomize(cell_number)

    def randomize(
        self,
        cell_number,
    ):
        self.x, self.y = np.random.random_integers(0, cell_number - 1, size=(2,))
        self.pos = Vector2(self.x, self.y)
