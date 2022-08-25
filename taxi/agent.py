import numpy as np
from collections import defaultdict
import pickle
import random

class Agent:

    def __init__(self, nA=6, eps0=1.0):
        """ Initialize the agent.

        Params
        ======
        - nA: number of actions available to the agent
        - eps0: epsilon hyperparameter of ε-greedy action selection
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.eps0 = eps0

    def select_action(self, state, eps):
        """ Given the state, select an action using ε-greedy. 

        Params
        ======
        - state: the current state of the environment
        - eps: eps parameter

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if random.random() > eps:
          return np.argmax(self.Q[state])
        return np.random.choice(self.nA)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.Q[state][action] += 1

    def save_policy(self, used_method):
      optimal_policy_estimate = dict()
      for state, Qs in self.Q.items():
        optimal_policy_estimate[state] = np.argmax(Qs)
      with open(f"learned_policies/learned_policy_from_{used_method}.pickle", "wb") as output_file:
        pickle.dump(optimal_policy_estimate, output_file)
