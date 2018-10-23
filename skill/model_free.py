import itertools
from collections import deque

import numpy as np

from skill.gridworld import Gridworld, GoalGridworld
from skill.replay_buffer import ReplayBuffer


class Trainer:
    def __init__(
            self,
            visit_func: callable,
            env: Gridworld,
            alpha: float = .001,
            gamma: float = .99,
    ):
        # rewards
        self.gamma = gamma
        self.alpha = alpha
        self.env = env
        self.R = np.zeros(env.nS)
        self.given_R = np.vectorize(visit_func)(np.arange(env.nS))

        # visitation frequency
        self.D = np.eye(env.nS)
        self.delD = np.zeros((env.nS, env.nS, env.nS))

        self.Q = 1e-5 * np.ones((env.nS, env.nA))
        self.delQ = np.zeros((env.nS, env.nA, env.nS))
        self.pi = None
        self.delPi = None

        self.P = env.transition_matrix
        self.I = np.eye(env.nS)
        self.start_states, = np.nonzero(env.isd)
        self.nS = env.nS
        self.nA = env.nA
        self.buffer = ReplayBuffer(maxlen=self.nS * 10)

    def iterate_array(self, rank: int, dim: int):
        if rank == 0:
            yield tuple()
        for i in range(dim):
            for tup in self.iterate_array(rank=rank-1, dim=dim):
                yield (i,) + tup

    def group_actions(self, A: np.ndarray, history: tuple, threshold: float=None):
        if threshold is None:
            threshold = 1. / self.nA
        if np.allclose(history , 0):
            return []
        groups = []
        for a, weight in enumerate(A[history]):
            if weight > threshold:
                for tail in self.group_actions(A, history[1:] + (a,)):

        # return [[a] + tail
        #         for a, b in enumerate(A[history]) if b > threshold
        #         for tail in self.group_actions(A, history[1:] + (a,))]



    def run_episode(self, env: GoalGridworld, Q: np.ndarray, A: np.ndarray,
                    epsilon: float = .01, alpha: float = .01,
                    gamma: float = .99):
        s1 = env.reset()
        actions = deque([env.nA] * len(A.shape),
                        maxlen=len(A.shape))
        G = 0
        for i in itertools.count():
            if np.random.random() < epsilon:
                a = env.action_space.sample()
            else:
                a = np.argmax(Q[s1])
            actions.append(a)
            A[tuple(actions)[:-1]] *= (1-alpha)
            A[tuple(actions)] += alpha

            s2, r, t, i = env.step(a)
            G += r
            Q[s1] += alpha * (r + gamma * np.max(Q[s2], axis=1) - Q[s1])
            if t:
                return G

    def train_goal(self, env: GoalGridworld, len_action_history: int, **kwargs):
        returns_queue = deque(maxlen=10)
        Q = np.zeros((env.nS, env.nA))
        A = np.zeros([env.nA] * (len_action_history + 1))
        while True:
            returns = self.run_episode(env=env, Q=Q, A=A, **kwargs)
            returns_queue.append(returns)
            if np.mean(returns_queue) == env.optimal_reward:
                action_groups = [ ]
                A =
                return Q

    def update(self):
        [(s1, s2, t), (g, _, _)] = self.buffer.sample(2)
        self.D = self.new_D(s1=s1, s2=s2, g=g)
        self.delPi = self.delPi()
        assert self.delPi.shape == (self.nS, self.nA, self.nS)
        self.delQ = self.new_delQ()
        self.delD = self.new_delD()
        self.Q = self.new_Q()
        self.R = self.new_R()

    def new_D(self, s1: int, s2: int, g: int):
        # s, a, s', g
        D = (self.D.reshape(1, 1, self.nS, self.nS))
        I = (self.I.reshape(self.nS, 1, 1, self.nS))
        P = (self.P.reshape(self.nS, self.nA, self.nS, 1))
        pi = (self.pi.reshape(self.nS, self.nA, 1, 1))
        return float(s1 == g) + D[s2]

    def delPi(self):
        # s, a_pi, a_Q, (s_r)
        I = (np.eye(self.nA).reshape(1, self.nA, self.nA))
        delQ = (self.delQ.reshape(self.nS, 1, self.nA, self.nS))
        pi = (self.pi.reshape(self.nS, self.nA, 1))
        dPi_dQ = (pi * (I - pi.transpose([0, 2, 1])))
        dPi_dQ = dPi_dQ.reshape(self.nS, self.nA, self.nA, 1)
        return np.sum(dPi_dQ * delQ, axis=2)

    def new_delD(self):
        # s_D, a, s', g, s_r
        D = (self.D.reshape(1, 1, self.nS, self.nS, 1).copy())
        P = (self.P.reshape(self.nS, self.nA, self.nS, 1, 1).copy())
        delD = (self.delD.reshape(1, 1, self.nS, self.nS, self.nS).copy())
        delPi = (self.delPi.reshape(self.nS, self.nA, 1, 1, self.nS).copy())
        pi = (self.pi.reshape(self.nS, self.nA, 1, 1, 1).copy())
        return np.sum(P * (delPi * D + pi * delD), axis=(1, 2))

    def new_delQ(self):
        # s_Q, a, s', a', s_r
        I = (self.I.reshape(self.nS, 1, 1, 1, self.nS))
        P = (self.P.reshape(self.nS, self.nA, self.nS, 1, 1))
        Q = (self.Q.reshape(1, 1, self.nS, self.nA, 1))
        delPi = (self.delPi.reshape(1, 1, self.nS, self.nA, self.nS))
        delQ = (self.delQ.reshape(1, 1, self.nS, self.nA, self.nS))
        pi = (self.pi.reshape(1, 1, self.nS, self.nA, 1))
        return np.sum(I + P * (delPi * Q + pi * delQ), axis=(2, 3))

    def new_Q(self):
        # s, a, s', (a')
        P = (self.P.reshape(self.nS, self.nA, self.nS))
        Q = (self.Q.reshape(1, 1, self.nS, self.nA))
        R = (self.R.reshape(self.nS, 1, 1))
        return np.sum(R + self.gamma * P * np.max(Q, axis=3), axis=2)

    def new_R(self):
        # s0, g, s_r
        R = self.R
        given_R = (self.given_R.reshape(1, self.nS, 1))
        alpha = self.alpha
        delD = self.delD
        start_states = self.start_states
        return R + alpha * np.sum(delD[start_states] * given_R, axis=(0, 1))
