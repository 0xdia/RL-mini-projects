from env import GridWorldEnv
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

env = GridWorldEnv(size=20)

model = DQN(
    "MultiInputPolicy",
    env,
    learning_rate=0.0001,
    buffer_size=1000000,
    learning_starts=50000,
    batch_size=32,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=10000,
    exploration_fraction=0.35,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    max_grad_norm=10,
    tensorboard_log="./tensorboard/",
)

model.learn(
    total_timesteps=int(1e7),
    log_interval=1000,
    tb_log_name="1e7",
    progress_bar=True,
)

model.save("dqn_model")
evaluate_policy(model, env, render=True, n_eval_episodes=50)
env.close()
