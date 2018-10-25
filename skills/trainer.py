# stdlib
from collections import defaultdict, deque
import itertools
import time
from typing import Iterable, List, Sequence, Sized, Union

# third party
import numpy as np
import plotly
import plotly.graph_objs as go

# first party
from skills.plot import plot
from skills.replay_buffer import ReplayBuffer


class Trainer:
    def __init__(
            self,
            env,
            alpha: float = .1,
            gamma: float = .99,
            epsilon: float = .1,
            action_window: int = 10,
            n_action_groups: int = 5,
            slack_factor: int = 1,
    ):
        self.action_window = min(action_window, env._max_episode_steps)
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
        for i in range(len(string)):
            for j in range(i + 2, len(string) + 1):
                self.A[tuple(string[i:j])] += 1

    def discounted_cumulative(self, rewards):
        return sum(r * self.gamma**i for i, r in enumerate(rewards))

    def step(self, actions: Iterable[int]):
        rewards = []
        terminal = False
        info = dict()
        for action in actions:
            s2, r, t, i = self.env.step(action)
            #debugging
            if self.render:
                print(actions)
                self.env.render()
                time.sleep(1.5 if t else .5)

            rewards.append(r)
            info.update(i)
            if t:
                terminal = True
                break
        return s2, rewards, terminal, info

    def run_episode(self, s1: int, Q: np.ndarray, action_groups: List):
        env = self.env
        epsilon = self.epsilon
        alpha = self.alpha
        gamma = self.gamma
        nA = Q.shape[-1]

        episode_rewards = []
        episode_actions = []
        for i in itertools.count():
            try:
                random = np.random.random()
                if random < epsilon:
                    a = np.random.randint(nA)
                else:
                    a = int(argmax(Q[s1]))
                if a >= self.nA:
                    actions = action_groups[a - self.nA]
                else:
                    actions = [a]

                # if all([
                # np.allclose(self.goal, (4, 2)),
                # np.allclose(self.s1, (1, 2)),
                # s1 == self.encode(*self.s1),
                # actions == (1, 1, 1),
                # self.env._elapsed_steps < 7,
                # ]):
                # self.render = True
                # import ipdb
                # ipdb.set_trace()

                s2, R, t, _ = self.step(actions)
                episode_actions.extend(actions)
                episode_rewards.extend(R)
                r = self.discounted_cumulative(R)
                Q[s1, a] += alpha * (
                    r + (not t) * gamma * np.max(Q[s2]) - Q[s1, a])
                # _Q = Q.copy()
                # _Q[s1, a] += alpha * (
                # r + (not t) * gamma * np.max(Q[s2]) - Q[s1, a])

                # Q[:] = _Q
                s1 = s2
                if t:
                    returns = self.discounted_cumulative(episode_rewards)
                    return episode_actions, returns
            except KeyboardInterrupt:
                print(self.gridworld.decode(self.gridworld.goal))
                self.plotQ(Q)
                import ipdb
                ipdb.set_trace()

    def train_goal(self):
        action_groups = sorted(self.A.keys(), key=lambda k: self.A[k])
        action_groups = action_groups[-self.n_action_groups:]
        returns_queue = deque([0] * 10, maxlen=10)
        env = self.env
        nA = self.nA + len(action_groups)
        Q = np.zeros((self.nS, nA))
        s1 = env.reset()
        self.gridworld.desc[self.gridworld.decode(s1)] = 'S'

        self.s1 = np.array(self.gridworld.decode(s1))
        self.goal = np.clip(self.s1 + np.array([3, 0]), np.zeros(2),
                            np.array(self.gridworld.desc.shape) - 1)
        goal = self.gridworld.encode(*self.goal)
        self.gridworld.set_goal(self.gridworld.encode(*self.goal))

        optimal_reward = self.optimal_reward(s1, goal)

        for i in itertools.count():
            actions, returns = self.run_episode(
                s1=s1, Q=Q, action_groups=action_groups)

            # reset
            env.reset()
            self.gridworld.s = s1

            returns_queue.append(returns)
            # print(returns, optimal_reward)
            if np.mean(returns_queue
                       ) >= optimal_reward * self.slack:  # * self.gamma**2:
                print('episodes', i, 'start', self.s1, 'goal', self.goal)
                self.gridworld.desc[self.s1] = '◻'
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
