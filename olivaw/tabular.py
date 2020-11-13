import copy
import collections
import random

import numpy as np


class _Agent:
    def __init__(self,
                 env,
                 nb_episodes=25000,
                 max_steps=200,
                 learning_rate=0.01,
                 gamma=0.99,
                 epsilon_greedy=None,
                 verbose=False):
        self.nb_episodes = nb_episodes
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.epsilon_greedy = epsilon_greedy

        self.verbose = verbose
        if self.verbose:
            print(f"There are {env.observation_space.n} possible states")
            print(f"And there are {env.action_space.n} possible actions")

        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
        self.env = env

    def learn(self):
        log_every = 1000
        mean_reward = []

        for episode in range(self.nb_episodes):
            state = self.env.reset()

            episode_reward = 0
            for step in range(self.max_steps):
                action = self.epsilon_greedy.sample(self.q_table, state, self.env)

                new_state, reward, done, info = self.env.step(action)
                self._update_rule(state, new_state, reward, action)
                episode_reward += reward

                if done:
                    break # end of episode

                state = new_state

            episode_reward /= (step + 1)
            mean_reward.append(episode_reward)
            self.epsilon_greedy.on_episode_end()

            if episode > 0 and episode % log_every == 0 and self.verbose:
                print(f"Episode {episode}/{total_episodes}, mean reward of last {log_every} episodes: {sum(mean_reward[-log_every:])/log_every}")

        return np.arange(self.nb_episodes), mean_reward

    def test(self, nb_episodes):
        rewards = []

        for episode in range(nb_episodes):
            state = self.env.reset()
            total_rewards = 0
            if self.verbose:
                print("****************************************************")
                print("EPISODE ", episode)

            for step in range(self.max_steps):
                if self.verbose:
                    self.env.render()
                # Take the action (index) that have the maximum expected future reward given that state
                action = self.q_table[state].argmax()
                new_state, reward, done, info = self.env.step(action)
                total_rewards += reward

                if done:
                    rewards.append(total_rewards)
                    #print ("Score", total_rewards)
                    break
                state = new_state
        self.env.close()
        print ("Score over time: " +  str(sum(rewards)/ nb_episodes))

    def _update_rule(self, state, new_state, reward, action):
        raise NotImplementedError


class QLearning(_Agent):
    def _update_rule(self, state, new_state, reward, action):
        residual = reward + self.gamma * np.max(self.q_table[new_state]) - self.q_table[state, action]
        self.q_table[state, action] = self.q_table[state, action] + self.learning_rate * (residual)


class Sarsa(_Agent):
    def _update_rule(self, state, new_state, reward, action):
        new_action = self.epsilon_greedy.sample(self.q_table, new_state, self.env)

        residual = reward + self.gamma * self.q_table[new_state, new_action] - self.q_table[state, action]
        self.q_table[state, action] = self.q_table[state, action] + self.learning_rate * (residual)


class _MonteCarlo(_Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_table_freq = copy.deepcopy(self.q_table)

    def learn(self):
        log_every = 1000
        mean_reward = []


        for episode in range(self.nb_episodes):
            state = self.env.reset()

            episode_stats = []
            episode_reward = 0
            for step in range(self.max_steps):
                action = self.epsilon_greedy.sample(self.q_table, state, self.env)

                new_state, reward, done, info = self.env.step(action)
                episode_stats.append((state, action, reward))

                episode_reward += reward

                if done:
                    break # end of episode

                state = new_state

            self._update_rule(episode_stats)

            episode_reward /= (step + 1)
            mean_reward.append(episode_reward)
            self.epsilon_greedy.on_episode_end()

            if episode > 0 and episode % log_every == 0 and self.verbose:
                print(f"Episode {episode}/{total_episodes}, mean reward of last {log_every} episodes: {sum(mean_reward[-log_every:])/log_every}")

        return np.arange(self.nb_episodes), mean_reward



class MonteCarloOnPolicy(_MonteCarlo):
    def _update_rule(self, episode_stats):
        G = collections.defaultdict(int)
        cum_rewards = 0

        for state, action, reward in episode_stats[::-1]:
            cum_rewards = reward + self.gamma * cum_rewards
            G[(state, action)] = cum_rewards

        for (state, action), reward in G.items():
            # The frequency table is there to avoid giving too much importance
            # to a (state, action) that occured many times
            q = self.q_table[state, action]
            self.q_table_freq[state, action] += 1
            n = self.q_table_freq[state, action]

            self.q_table[state, action] = q * n / (n + 1) + G[(state, action)] / (n + 1)
