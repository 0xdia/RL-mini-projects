import numpy as np
from collections import defaultdict
import pickle
import random

class Agent:

    def __init__(self, nA=6, eps0=1.0, gamma=1.0, alpha=0.2):
        """ Initialize the agent.

        Params
        ======
        - nA: number of actions available to the agent
        - eps0: epsilon hyperparameter of ε-greedy action selection
        - gamma: dicount factor
        - alpha: alpha hyperparameter
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.ones(self.nA))
        self.eps0 = eps0
        self.eps = eps0
        self.gamma = gamma
        self.alpha = alpha
      
    def update_eps(self, episode_num):
      """ Update ε following the formula:
              ε <- ε0 / (1 + episode_num * 0.2)

      Params
      ======
      - episode_num: the number of the current episode
      """
      self.eps = self.eps0 / (1 + 0.2 * episode_num)
     
    def select_action(self, state, learning_is_frozen=False):
        """ Given the state, select an action.
            If learning_is_frozen, the best action is selected.
            Else, ε-greedy is used. 

        Params
        ======
        - state: the current state of the environment
        - eps: eps parameter
        - learning_is_frozen: whether learning is frozen (testing mode)

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if learning_is_frozen or random.random() > self.eps:
          return np.argmax(self.Q[state])
        return np.random.choice(self.nA)

    def update_Q_sarsa(self, state, action, reward, next_state, next_action):
      """ Update the action-value following Sarsa method.

      Params
      ======
      - state: the previous state of the environment
      - action: the agent's previous choice of action
      - reward: last reward received
      - next_state: the current state of the environment
      - next_action: the picked action to be performed next
      """
      expected_return = reward + self.gamma * self.Q[next_state][next_action] 
      self.Q[state][action] += self.alpha * (expected_return - self.Q[state][action])

    def update_Q_sarsamax(self, state, action, reward, next_state):
      """ Update the action-value following Q-learning method.

      Params
      ======
      - state: the previous state of the environment
      - action: the agent's previous choice of action
      - reward: last reward received
      - next_state: the current state of the environment
      """
      expected_return = reward + self.gamma * np.max(self.Q[next_state])
      self.Q[state][action] += self.alpha * (expected_return - self.Q[state][action])

    def update_Q_expected_sarsa(self, state, action, reward, next_state):
      """ Update the action-value following Expected Sarsa method.

      Params
      ======
      - state: the previous state of the environment
      - action: the agent's previous choice of action
      - reward: last reward received
      - next_state: the current state of the environment
      """
      state_policy = np.ones(self.nA) * self.eps / self.nA
      state_policy[np.argmax(self.Q[next_state])] = 1 - self.eps + self.eps / self.nA
      expected_return = reward + self.gamma * np.dot(state_policy, self.Q[next_state])
      self.Q[state][action] += self.alpha * (expected_return - self.Q[state][action])
      
    def step(self, state, action, reward, next_state, next_action, done, method, freeze_learning=False):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - method: TD method ("sarsa", "q_learning" or "expected_sarsa")
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - next_action: the picked action to be performed next
        - done: whether the episode is complete (True or False)
        - freeze_learning: whether to freeze learning (testing mode)
        """
        if freeze_learning:
          return  
        assert method in ["sarsa", "q_learning", "expected_sarsa"]
        if method == "sarsa":
          self.update_Q_sarsa(state, action, reward, next_state, next_action) 
        elif method ==  "q_learning":
          self.update_Q_sarsamax(state, action, reward, next_state)
        else:
          self.update_Q_expected_sarsa(state, action, reward, next_state)

    def get_policy(self):
      optimal_policy_estimate = dict()
      for state, Qs in self.Q.items():
        optimal_policy_estimate[state] = np.argmax(Qs) 
      return optimal_policy_estimate
      
    def save_policy(self, used_method):
      policy = self.get_policy()
      with open(f"learned_policies/learned_policy_from_{used_method}.pickle", "wb") as output_file:
        pickle.dump(policy, output_file)
