import itertools
from collections import deque

import numpy as np
import time

from skills.gridworld import GoalGridworld
from skills.replay_buffer import ReplayBuffer
import gym

from skills.util import get_wrapped_attr


class Trainer:
    def __init__(
            self,
            env: gym.Env,
            len_action_history: int,
            alpha: float = .001,
            gamma: float = .99,
            epsilon: float = .01,
    ):
        self.len_action_history = len_action_history
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.env = env
        self.nS = get_wrapped_attr(env, 'nS')
        self.nA = get_wrapped_attr(env, 'nA')
        self.optimal_reward = get_wrapped_attr(env, 'optimal_reward')
        self.buffer = ReplayBuffer(maxlen=self.nS * 10)

    def iterate_array(self, rank: int, dim: int):
        if rank == 0:
            yield tuple()
        for i in range(dim):
            for tup in self.iterate_array(rank=rank - 1, dim=dim):
                yield (i, ) + tup

    def group_actions(self,
                      A: np.ndarray,
                      history: tuple,
                      threshold: float = None):
        if threshold is None:
            threshold = 1. / self.nA
        if np.allclose(history, 0):
            return []
        groups = []
        for a, weight in enumerate(A[history]):
            if weight > threshold:
                for tail in self.group_actions(A, history[1:] + (a, )):
                    groups.append([a] + tail)
        return groups

        # return [[a] + tail
        #         for a, b in enumerate(A[history]) if b > threshold
        #         for tail in self.group_actions(A, history[1:] + (a,))]

    def run_episode(
            self,
            Q: np.ndarray,
            A: np.ndarray,
    ):
        env = self.env
        epsilon = self.epsilon
        alpha = self.alpha
        gamma = self.gamma
        s1 = env.reset()
        actions = deque(
            [self.nA] * len(
                A.shape),  # the nA^th action is the special stop action
            maxlen=len(A.shape))
        G = 0
        for i in itertools.count():
            random = np.random.random()
            if random < epsilon:
                a = env.action_space.sample()
            else:
                a = np.argmax(Q[s1])
            actions.append(a)
            A[tuple(actions)[:-1]] *= (1 - alpha)
            A[tuple(actions)] += alpha

            env.render()
            time.sleep(.1)
            s2, r, t, i = env.step(a)
            G += r
            Q[s1] += alpha * (r + gamma * np.max(Q[s2]) - Q[s1])
            if t:
                return G

    def train_goal(self):
        returns_queue = deque(maxlen=10)
        len_action_history = self.len_action_history
        env = self.env
        Q = np.zeros((self.nS, self.nA))
        A = np.zeros([self.nA + 1] * (len_action_history + 1))
        while True:
            returns = self.run_episode(Q=Q, A=A)
            print(returns)
            returns_queue.append(returns)
            if np.mean(returns_queue) == self.optimal_reward:
                action_groups = self.group_actions(
                    A=A, history=tuple([0] * len_action_history))
                import ipdb
                ipdb.set_trace()
                return Q, action_groups

    def train(self):
        for _ in itertools.count():
            self.train_goal()
