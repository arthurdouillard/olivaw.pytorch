import random

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
import numpy as np


class ExperienceReplay:
    def __init__(self, replay_size, frame_size, stack_size):
        # * state, action, reward, next_state, done --> 5 objects
        self.buffer_state = np.empty((replay_size, stack_size, *frame_size), dtype=np.uint8)
        self.buffer_action = np.empty((replay_size,), dtype=np.uint8)
        self.buffer_reward = np.empty((replay_size,), dtype=np.uint8)
        self.buffer_next_state = np.empty((replay_size, stack_size, *frame_size), dtype=np.uint8)
        self.buffer_done = np.empty((replay_size,), dtype=bool)

        self.end = 0
        self.filed_size = 0
        self.max_size = replay_size

        self.trsfs = lambda x: torch.tensor(x / 255).float()

    def add(self, exp):
        self.buffer_state[self.end] = exp[0]
        self.buffer_action[self.end] = exp[1]
        self.buffer_reward[self.end] = exp[2]
        self.buffer_next_state[self.end] = exp[3]
        self.buffer_done[self.end] = exp[4]

        self.filed_size = min(self.filed_size + 1, self.max_size)
        self.end = (self.end + 1) % self.max_size

    def sample(self, batch_size):
        indexes = np.random.choice(
            self.filed_size,
            size=batch_size,
            replace=False
        )
        return {
            "state": self.trsfs(self.buffer_state[indexes]),
            "action": torch.tensor(self.buffer_action[indexes]).long(),
            "reward": torch.tensor(self.buffer_reward[indexes]).long(),
            "next_state": self.trsfs(self.buffer_next_state[indexes]),
            "done": torch.tensor(self.buffer_done[indexes]).long()
        }

    def prefill(self, stacked_frames, action_size, pretrain_length, env):
        done = True
        for step in range(pretrain_length):
            if done:
                state = env.reset()
                stacked_frames.on_new_episode(state)
                state = stacked_frames.get()
                done = False

            action = np.random.randint(1, action_size)
            next_state, reward, done, _ = env.step(action)

            stacked_frames.on_new_step(next_state)
            next_state = stacked_frames.get()

            if done:
                next_state = np.zeros((210, 160, 3))  # to fix
                stacked_frames.on_new_episode(next_state)
                next_state = stacked_frames.get()
            self.add((state, action, reward, next_state, done))

            state = next_state

        print(f"Prefilled {self.end} frames.")


class PrioritizedExperienceReplay(ExperienceReplay):
    def __init__(self, *args, alpha=0.6, beta=0.4, eta=0.00025/4, epsilon=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.sumtree = SumTree(self.max_size)

        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.epsilon = epsilon

    def add(self, exp):
        super().add(exp)
        self.sumtree.add(self.end)

    def update_transition(self, errors):
        """Line 12 of Algo 1 of Schaul et al. 2016"""
        self.sumtree.update_priorities(self._latest_indexes, errors)

    def sample(self, batch_size):
        """Line 9-10 of Algo 1 of Schaul et al. 2016"""
        indexes, priorities = self.sumtree.sample(batch_size)

        p = (priorities + self.epsilon) ** self.alpha
        p = p / np.sum(p)
        w = (self.max_size * p) ** (-self.beta)
        is_weights = w / w.max()

        self._latest_indexes = indexes

        self.beta = min(1.0, self.beta + self.eta)

        return {
            "state": self.trsfs(self.buffer_state[indexes]),
            "action": torch.tensor(self.buffer_action[indexes]).long(),
            "reward": torch.tensor(self.buffer_reward[indexes]).long(),
            "next_state": self.trsfs(self.buffer_next_state[indexes]),
            "done": torch.tensor(self.buffer_done[indexes]).long(),
            "is_weights": torch.tensor(is_weights)
        }


class SumTree:
    """Data Structure for efficient sampling of indexes with different priorities.

    This SumTree stores only the indexes of the Experience Replay memory.
    Each index has an associated priority, reflecting its importance, and thus
    its sampling *priority*.

    Two internal arrays are used:
    - `self.indexes` storing all indexes in a contiguous fashion
    - `self.priorities` storing the priorities in a tree stored in an array

    For the second, the actual tree, we store it in an array for efficiency.
    Given a node at position `i`, its left child is at `2 * i + 1` and right child at `2 * i + 2`.
    The root is at 0.

    Only leaves are storing the actual indexes, thus all other nodes are kinda "virtual",
    but we need them in order to walk through the tree deep first.
    """
    def __init__(self, replay_size):
        self.indexes = np.zeros((replay_size,), dtype=np.uint32)
        self.priorities = np.zeros((2 * replay_size - 1,), dtype=np.float32)

        self.max_size = replay_size
        self.end = 0

    def sample(self, batch_size):
        max_priority = self.priorities[0]
        indexes = np.zeros((batch_size,), dtype=np.uint32)
        priorities = np.empty((batch_size,))

        priority_interval = max_priority // batch_size

        for batch_index in range(batch_size):
            sampled_priority = random.uniform(
                batch_index * priority_interval,
                (batch_index + 1) * priority_interval
            )

            tree_pos = self._retrieve(0, sampled_priority)  # 0 -> starting from the root
            priorities[batch_index] = self.priorities[tree_pos]

            index = (tree_pos + 1) - self.max_size  # +1 because tree_pos starts at 1, not 0
            indexes[batch_index] = self.indexes[index]

        return indexes.astype(np.uint32), priorities

    def _retrieve(self, pos, sampled_priority):
        left_pos = 2 * pos + 1
        right_pos = left_pos + 1

        if left_pos >= len(self.priorities):
            return pos  # terminal condition of the recursion
        elif self.priorities[left_pos] > sampled_priority:
            return self._retrieve(left_pos, sampled_priority)
        return self._retrieve(right_pos, sampled_priority - self.priorities[left_pos])

    def update_priorities(self, indexes, errors):
        tree_positions = indexes + self.max_size - 1
        self.priorities[tree_positions] = np.abs(errors)

    def add(self, index, priority=None):
        if priority is None:  # If no initial priority is given, set to max
            priority = self.priorities[0]

        self.indexes[self.end] = index
        tree_pos = self.end + self.max_size - 1

        previous_priority = self.priorities[tree_pos]
        self.priorities[tree_pos] = priority

        self._ascend_propagate(tree_pos, priority - previous_priority)

        self.end = (self.end + 1) % self.max_size

    def _ascend_propagate(self, pos, priority_change):
        pos = (pos - 1) // 2
        self.priorities[pos] += priority_change
        if pos != 0:  # Until we are at tree root
            self._ascend_propagate(pos, priority_change)
