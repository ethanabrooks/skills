import itertools
from collections import deque, defaultdict

import numpy as np
from typing import Iterable, Union, Sequence, List, Sized

from skills.replay_buffer import ReplayBuffer

from skills.plot import plot


class SubstrCounter:
    def __init__(self, alphabet_or_size: Union[str, int], minlen: int,
                 maxlen: int):
        if isinstance(alphabet_or_size, str):
            alphabet = alphabet_or_size
        elif isinstance(alphabet_or_size, int):
            alphabet = range(alphabet_or_size)
        else:
            raise TypeError('alphabet_or_size must be a string or an int')
        self.indices = {a: i for i, a in enumerate(alphabet)}
        self.minlen = minlen
        self.arrays = [
            np.zeros((len(self.alphabet), ) * i
                     for i in range(minlen + 1, maxlen + 1))
        ]
        self.heap = []

    def get_array(self, lenkey: int) -> np.ndarray:
        return self.arrays[lenkey - self.minlen]

    def __getitem__(self, key: Sized) -> int:
        return self.get_array(len(key))[tuple(key)]

    def __setitem__(self, key: Sized, value: int):
        self.get_array(len(key))[tuple(key)] = value


class Trainer:
    def __init__(
            self,
            env,
            alpha: float = .1,
            gamma: float = .99,
            epsilon: float = .1,
            action_window: int = 10,
            n_action_groups: int = 5,
    ):
        self.action_window = min(action_window, env._max_episode_steps)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.env = env
        self.gridworld = env.unwrapped
        self.nS = self.gridworld.nS
        self.nA = self.stop_action = self.gridworld.nA
        self.buffer = ReplayBuffer(maxlen=self.nS * 10)
        self.A = defaultdict(lambda: 0)
        self.n_action_groups = n_action_groups

    def iterate_array(self, rank: int, dim: int):
        if rank == 0:
            yield tuple()
        for i in range(dim):
            for tup in self.iterate_array(rank=rank - 1, dim=dim):
                yield (i, ) + tup

    def count_substrs(self, string: Sequence):
        for i in range(len(string)):
            for j in range(i + 2, len(string) + 1):
                self.A[tuple(string[i:j])] += 1

        # return [[a] + tail
        #         for a, b in enumerate(A[history]) if b > threshold
        #         for tail in self.count_substrs(A, history[1:] + (a,))]

    def run_episode(self, s1: int, Q: np.ndarray, action_groups: List):
        env = self.env
        epsilon = self.epsilon
        alpha = self.alpha
        gamma = self.gamma
        nA = Q.shape[-1]

        rewards = []
        actions = []
        for i in itertools.count():
            try:
                random = np.random.random()
                if random < epsilon:
                    a = np.random.randint(nA)
                else:
                    a = int(argmax(Q[s1]))
                if a >= self.nA:
                    A = action_groups[a - self.nA]
                else:
                    A = [a]
                for a in A:
                    actions.append(a)
                    s2, r, t, i = env.step(a)
                    rewards.append(r)
                    Q[s1, a] += alpha * (r + gamma * np.max(Q[s2]) - Q[s1, a])
                    s1 = s2
                    if t:
                        Q[s2] += alpha * r
                        returns = sum(
                            r * self.gamma**i for i, r in enumerate(rewards))
                        return actions, returns
            except KeyboardInterrupt:
                print(self.gridworld.decode(self.gridworld.goal))
                plot(Q.max(axis=1).reshape(self.gridworld.desc.shape))
                import ipdb
                ipdb.set_trace()

    def train_goal(self):
        action_groups = sorted(self.A.keys(), key=lambda k: self.A[k])
        action_groups = action_groups[-self.n_action_groups:]
        returns_queue = deque(maxlen=10)
        env = self.env
        nA = self.nA + len(action_groups)
        Q = np.zeros((self.nS, nA))
        s1 = env.reset()
        optimal_reward = self.optimal_reward(s1)

        _s1 = np.array(self.gridworld.decode(s1))
        _goal = np.clip(s1 + np.array([3, 0]), 0, 9)
        self.gridworld.set_goal(self.gridworld.encode(*_goal))

        for i in itertools.count():
            actions, returns = self.run_episode(
                s1=s1, Q=Q, action_groups=action_groups)

            # reset
            env.reset()
            self.gridworld.s = s1

            returns_queue.append(returns)
            if np.allclose(returns_queue, optimal_reward):
                # plot(Q.max(axis=1).reshape(self.gridworld.desc.shape))
                print()
                print(i, self.gridworld.goal)
                return actions

    def train(self):
        for _ in itertools.count():
            actions = self.train_goal()
            self.count_substrs(actions)

    def optimal_reward(self, s1: int):
        s1 = np.array(self.gridworld.decode(s1))
        goal = np.array(self.gridworld.decode(self.gridworld.goal))
        min_steps = np.sum(np.abs(goal - s1))
        return self.gamma**(min_steps - 1)

    def plotQ(self, Q):
        env = self.env
        Q_reshaped = np.flip(
            Q.reshape(env.unwrapped.desc.shape + (self.nA, )), axis=0)
        layout = {
            f'yaxis{i + 1}': dict(
                ticktext=tuple(reversed(range(self.nS))),
                tickvals=tuple(range(self.nS)))
            for i in range(self.nA)
        }

        plot(
            Q_reshaped.transpose((2, 0, 1)),
            layout=layout,
            subplot_titles=env.unwrapped.action_strings)
        import ipdb
        ipdb.set_trace()


def argmax(x: np.ndarray, axis=-1) -> np.ndarray:
    max_x = x == np.max(x, axis=axis)

    def max_index(row):
        nonzero, = np.nonzero(row)
        return np.random.choice(nonzero)

    return np.apply_along_axis(max_index, axis, max_x)
