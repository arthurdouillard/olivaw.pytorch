import random

import numpy as np
import torch


class EpsilonGreedy:
    def __init__(self, epsilon=1.0, min_epsilon=0.001, decay=0.01):
        self.epsilon = epsilon
        self.max_epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay

        self._episode = 0

    def update(self):
        self._episode += 1
        self.epsilon =  self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay * self._episode)

    def sample(self, q_table, state, env, only_greedy=False):
        if random.random() > self.epsilon or only_greedy:
            return q_table[state].argmax()
        return env.action_space.sample()


class DeepEpsilonGreedy(EpsilonGreedy):
    def sample(self, q_net, state, env, only_greedy=False):
        if random.random() > self.epsilon or only_greedy:
            if len(state.shape) == 3:
                state = state[None]
            state = state.to(q_net.device)
            with torch.no_grad():
                is_training = q_net.training
                q_net.eval()
                action = q_net(state)["qvalues"].argmax()
                q_net.train(is_training)
                return action
        return env.action_space.sample()
