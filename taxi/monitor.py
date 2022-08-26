from collections import deque
import sys
import math
import numpy as np

def interact(env, agent, num_episodes=20000, method="expected_sarsa", window=100, mode="train", save_policy=False):
    """ Monitor agent's performance.
    
    Params
    ======
    - env: instance of OpenAI Gym's Taxi-v3 environment
    - agent: instance of class Agent (see Agent.py for details)
    - num_episodes: number of episodes of agent-environment interaction
    - method: reinforcement learning method
    - window: number of episodes to consider when calculating average rewards
    - mode: training or testing mode
    - save_policy: whether to save the learned optimal policy estimate
    
    Returns
    =======
    - avg_rewards: deque containing average rewards
    - best_avg_reward: largest value in the avg_rewards deque
    """
    # initialize average rewards
    avg_rewards = deque(maxlen=num_episodes)
    # initialize best average reward
    best_avg_reward = -math.inf
    # initialize monitor for most recent rewards
    samp_rewards = deque(maxlen=window)
    # for each episode
    for i_episode in range(1, num_episodes+1):
        # begin the episode
        state = env.reset()
        # the agent selectd an action
        action = agent.select_action(state, learning_is_frozen=(mode=="test"))
        # update ε
        agent.update_eps(i_episode)
        # initialize the sampled reward
        samp_reward = 0
        while True:
            # agent performs the selected action
            next_state, reward, done, _ = env.step(action)
            # the agent selects an action for the next time step
            next_action = agent.select_action(next_state)
            # agent performs internal updates based on sampled experience
            agent.step(state, action, reward, next_state, next_action, done, method, freeze_learning=(mode=="test"))
            # update the sampled reward
            samp_reward += reward
            # update the state (s <- s') to next time step
            state = next_state
            # update the action (a <- a') to next time step
            action = next_action
            if done:
                # save final sampled reward
                samp_rewards.append(samp_reward)
                break
        if (i_episode >= 100):
          # get average reward from last 100 episodes
          avg_reward = np.mean(samp_rewards)
          # append to deque
          avg_rewards.append(avg_reward)
          # update best average reward
          if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
        if mode=="train":
          # monitor progress
          print("\rEpisode {}/{} || Best average reward {}".format(i_episode, num_episodes, best_avg_reward), end="")
          sys.stdout.flush()
          # check if task is solved (according to OpenAI Gym)
          if best_avg_reward >= 9.7:
            print('\nEnvironment solved in {} episodes.'.format(i_episode), end="")
            break
          if i_episode == num_episodes: print('\n')
    if save_policy:
      agent.save_policy(method)
    return np.mean(samp_rewards) if mode == "test" else (avg_rewards, best_avg_reward)