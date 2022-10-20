from pygame.math import Vector2
import numpy as np


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
