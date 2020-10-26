import random

import numpy as np


class EpsilonGreedy:
    def __init__(self, epsilon=1.0, max_epsilon=1.0, min_epsilon=0.001, decay=0.01):
        self.epsilon = epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay

        self._episode = 0

    def on_episode_end(self):
        self._episode += 1
        self.epsilon =  self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay * self._episode)

    def sample(self, q_table, state, env, only_greedy=False):
        if random.random() > self.epsilon or only_greedy:
            return q_table[state].argmax()
        return env.action_space.sample()
