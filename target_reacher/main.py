import matplotlib.pyplot as plt
from monitor import monitor

scores = monitor(num_episodes=500, train_mode=True, load_model=False, save_model=True)
plt.plot(scores)
plt.show()