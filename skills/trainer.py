# stdlib
from collections import defaultdict, deque
import itertools
import time
from typing import Iterable, List, Sequence, Sized

import sys
# third party
import numpy as np
import plotly
import plotly.graph_objs as go

# first party
from skills.replay_buffer import ReplayBuffer


class Trainer:
    def __init__(
            self,
            env,
            alpha: float = .1,
            gamma: float = .99,
            epsilon: float = .1,
            n_action_groups: int = 5,
            slack_factor: int = 1,
    ):
        self.gamma = gamma
        self.slack = gamma**slack_factor
        self.alpha = alpha
        self.epsilon = epsilon
        self.env = env
        self.gridworld = env.unwrapped
        self.nS = self.gridworld.nS
        self.nA = self.stop_action = self.gridworld.nA
        self.buffer = ReplayBuffer(maxlen=self.nS * 10)
        self.A = defaultdict(lambda: 0)
        self.n_action_groups = n_action_groups
        self.render = False

    def count_substrs(self, string: Sequence):
        min_len = 2
        for i in range(max(0, len(string) - min_len)):
            for j in range(i + min_len, len(string)):
                self.A[tuple(string[i:j])] += 1

    def discounted_cumulative(self, rewards):
        return sum(r * self.gamma**i for i, r in enumerate(rewards))

    def run_episode(self, s1: int, Q: np.ndarray, actions: List):
        epsilon = self.epsilon
        alpha = self.alpha
        gamma = self.gamma
        nA = len(actions)

        episode_states = []
        episode_actions = []
        episode_rewards = []
        #debugging
        if self.render:
            self.env.render()
            time.sleep(.1)
        for _ in itertools.count():
            random = np.random.random()
            if random < epsilon:
                a = np.random.randint(nA)
            else:
                a = int(argmax(Q[s1]))

            for action in actions[a]:
                episode_actions.append(action)
                episode_states.append(s1)
                s2, r, t, _ = self.env.step(action)
                if self.render:
                    self.env.render()
                    time.sleep(1.5 if t else .1)
                episode_rewards.append(r)

                for j in range(len(episode_actions), 0, -1):
                    action_group = tuple(episode_actions[-j:])
                    if action_group in actions:
                        a = actions.index(action_group)
                        Q[s1, a] += alpha * (
                            r + (not t) * gamma**j * np.max(Q[s2]) - Q[s1, a])
                s1 = s2
                if t:
                    returns = self.discounted_cumulative(episode_rewards)
                    if returns > 0:
                        print(returns, end=' ')
                        sys.stdout.flush()
                    return episode_actions, returns

    def train_goal(self):
        def time_saved(action_seq):
            return self.A[action_seq] * (len(action_seq) - 1)

        action_groups = list(sorted(self.A.keys(), key=time_saved))
        print(
            *sorted([(time_saved(k), ' '.join(
                [self.gridworld.action_strings[s] for s in k]))
                     for k in self.A]),
            sep='\n')

        action_groups = action_groups[-self.n_action_groups:]
        returns_queue = deque([0] * 8, maxlen=8)
        env = self.env
        nA = self.nA + len(action_groups)
        Q = np.zeros((self.nS, nA))
        s1 = env.reset()
        self.gridworld.desc[self.gridworld.decode(s1)] = 'S'

        self.s1 = np.array(self.gridworld.decode(s1))
        # self.goal = self.gridworld.decode(
        # self.gridworld.observation_space.sample())
        # offset = np.array([np.random.randint(-2, 2), 0])
        # offset = np.array([2, 0])
        _goal = self.gridworld.sample_goal()
        self.goal = self.gridworld.decode(_goal)

        goal = self.gridworld.encode(*self.goal)
        self.gridworld.set_goal(self.gridworld.encode(*self.goal))

        optimal_reward = self.optimal_reward(s1, goal)
        env.render()

        for i in itertools.count():
            actions = [(a, ) for a in range(self.nA)] + action_groups
            actions, returns = self.run_episode(s1=s1, Q=Q, actions=actions)

            # reset
            env.reset()
            self.gridworld.s = s1

            returns_queue.append(returns)
            # print(returns, optimal_reward)
            if np.mean(returns_queue
                       ) >= optimal_reward * self.slack:  # * self.gamma**2:
                self.gridworld.desc[tuple(self.s1)] = '_'
                print('\nfinal actions:', end=' ')
                print(' '.join(
                    [self.gridworld.action_strings[a] for a in actions]))
                return actions, i

    def train(self, iterations: int = 20, baseline: bool = False):
        times = []
        for i in range(iterations):
            actions, time_to_train = self.train_goal()
            times.append(time_to_train)
            if not baseline:
                self.count_substrs(actions)
        return times

    def optimal_reward(self, s1: int, goal: int):
        s1 = np.array(self.gridworld.decode(s1))
        goal = np.array(self.gridworld.decode(goal))
        min_steps = np.sum(np.abs(goal - s1))
        return self.gamma**min_steps

    def decode(self, s):
        return self.gridworld.decode(s)

    def encode(self, *s):
        return self.gridworld.encode(*s)

    def plotQ(self, Q):
        y, x = zip(self.s1, self.goal)
        matrix = Q.max(axis=1).reshape(self.gridworld.desc.shape)
        markers = go.Scatter(
            x=x,
            y=y,
            mode='text',
            text=['s1', 'goal'],
            textposition='middle center')

        q_values = go.Heatmap(z=matrix, colorscale='Viridis')
        plotly.offline.plot(
            [markers, q_values],
            auto_open=True,
        )
        # env = self.env
        # Q_reshaped = np.flip(
        # Q.reshape(env.unwrapped.desc.shape + (self.nA, )), axis=0)
        # layout = {
        # f'yaxis{i + 1}': dict(
        # ticktext=tuple(reversed(range(self.nS))),
        # tickvals=tuple(range(self.nS)))
        # for i in range(self.nA)
        # }

        # plot(
        # Q_reshaped.transpose((2, 0, 1)),
        # layout=layout,
        # subplot_titles=env.unwrapped.action_strings)
        # import ipdb
        # ipdb.set_trace()


def argmax(x: np.ndarray, axis=-1) -> np.ndarray:
    max_x = x == np.max(x, axis=axis)

    def max_index(row):
        nonzero, = np.nonzero(row)
        return np.random.choice(nonzero)

    return np.apply_along_axis(max_index, axis, max_x)
