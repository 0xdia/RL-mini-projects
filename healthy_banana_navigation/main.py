from unityagents import UnityEnvironment
import matplotlib.pyplot as plt
from monitor import run

env = UnityEnvironment(file_name="./bananas_env/Banana.x86_64")
scores, accuracy = run(env, 100, load_model=False, path='./pr_qnetwork', save_model=False, train_mode=True)
print(f"Best score: {max(scores)}")
plt.plot(accuracy)
plt.show()
