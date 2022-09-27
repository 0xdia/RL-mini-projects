import numpy as np
from collections import deque
from unityagents import UnityEnvironment

from agent import Agent

seed = 1337

def monitor(num_episodes=1500, train_mode=True, load_model=False, save_model=True):
    env = UnityEnvironment(file_name='./reacher_env/Reacher.x86_64') 

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=train_mode)[brain_name]    
    action_size = brain.vector_action_space_size
    state_size = env_info.vector_observations.shape[1]

    agent = Agent(state_size, action_size, -1., 1., load_model, seed)

    scores_deque = deque(maxlen=50)
    scores = []
    for episode in range(1, num_episodes+1):
        score = 0
        env_info = env.reset(train_mode=train_mode)[brain_name]    
        state = env_info.vector_observations[0]
        while True:
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            score += reward
            if train_mode:
                agent.step(reward, next_state, done)
            if done:
                break
            state = next_state
        scores_deque.append(score)
        scores.append(score)
        if episode % 50 == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_deque)))
    env.close()
    if save_model:
        agent.save_model()
    return scores 
