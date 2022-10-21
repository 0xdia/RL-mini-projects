import game
import snake


NUMBER_OF_CELLS = 10
NUMBER_OF_SNAKES = 1

snakes = snake.Snake(NUMBER_OF_CELLS)
env = game.GameGymEnv(snakes, NUMBER_OF_CELLS)

try:
    snakes.learn(env)
finally:
    print(snakes.evaluate(env, 20))
