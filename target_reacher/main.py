import matplotlib.pyplot as plt
from monitor import monitor

scores = monitor(train_mode=True, load_model=False, save_model=True)
plt.plot(scores)
plt.show()