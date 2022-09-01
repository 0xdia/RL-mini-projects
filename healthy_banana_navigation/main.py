from unityagents import UnityEnvironment
import matplotlib.pyplot as plt
from monitor import run

env = UnityEnvironment(file_name="./bananas_env/Banana.x86_64")
scores = run(env, 10, load_model=True, path='./qnetwork', save_model=False, train_mode=False)
print(f"Best score: {max(scores)}")
plt.plot(scores)
plt.show()
