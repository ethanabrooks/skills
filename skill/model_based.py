import itertools

import numpy as np

from skill.gridworld import Gridworld
from skill.util import softmax


class RewardOptimizer:
    def __init__(self,
                 visit_func: callable,
                 env: Gridworld,
                 alpha: float = .001,
                 gamma: float = .99, ):
        # rewards
        self.gamma = gamma
        self.alpha = alpha
        self.env = env
        self.R = np.zeros(env.nS)
        self.given_R = np.vectorize(visit_func)(np.arange(env.nS))

        # visitation frequency
        self.D = np.eye(env.nS)
        self.delD = np.zeros((env.nS, env.nS, env.nS))

        # Q
        self.Q = 1e-5 * np.ones((env.nS, env.nA))
        self.delQ = np.zeros((env.nS, env.nA, env.nS))

        self.P = env.transition_matrix
        self.I = np.eye(env.nS)
        self.start_states, = np.nonzero(env.isd)
        self.nS = env.nS
        self.nA = env.nA

    def train(self,
              delta: float = .001):
        # train loop
        ER = None
        for i in itertools.count():
            pi = softmax(self.Q, axis=1)
            self.D = self.new_D(pi)
            delPi = self.delPi(pi)
            assert delPi.shape == (self.nS, self.nA, self.nS)
            self.delQ = self.new_delQ(delPi=delPi, pi=pi)
            self.delD = self.new_delD(delPi=delPi, pi=pi)
            self.Q = self.new_Q()
            self.R = self.new_R()
            new_ER = np.sum(self.D * self.given_R)

            for array in [pi, self.D, delPi, self.delQ, self.delD, self.Q, self.R]:
                assert not np.any(np.isnan(array))

            if np.any(self.R):
                a_chars = np.array(tuple(self.env.action_strings))
                policy_string = a_chars[np.argmax(pi,
                                                  axis=1).reshape(self.env.desc.shape)]
                for r in policy_string:
                    print(''.join(r))

                # for i, r in enumerate(_R):
                # print(i, '|' * int(r))
                # print(Q.squeeze())
                # print(D.squeeze())
                # print('new', new_ER)
                # print('diff', new_ER - ER)
                print()

            if ER is not None and np.abs(new_ER - ER) < delta:
                return
            ER = new_ER

    def new_D(self, pi: np.ndarray):
        # s, a, s', g
        D = (self.D.reshape(1, 1, self.nS, self.nS))
        I = (self.I.reshape(self.nS, 1, 1, self.nS))
        P = (self.P.reshape(self.nS, self.nA, self.nS, 1))
        pi = (pi.reshape(self.nS, self.nA, 1, 1))
        return np.sum(I + pi * P * D, axis=(1, 2))

    def delPi(self, pi: np.ndarray):
        # s, a_pi, a_Q, (s_r)
        I = (np.eye(self.nA).reshape(1, self.nA, self.nA))
        delQ = (self.delQ.reshape(self.nS, 1, self.nA, self.nS))
        pi = (pi.reshape(self.nS, self.nA, 1))
        dPi_dQ = (pi * (I - pi.transpose([0, 2, 1])))
        dPi_dQ = dPi_dQ.reshape(self.nS, self.nA, self.nA, 1)
        return np.sum(dPi_dQ * delQ, axis=2)

    def new_delD(self, delPi: np.ndarray, pi: np.ndarray):
        # s_D, a, s', g, s_r
        D = (self.D.reshape(1, 1, self.nS, self.nS, 1).copy())
        P = (self.P.reshape(self.nS, self.nA, self.nS, 1, 1).copy())
        delD = (self.delD.reshape(1, 1, self.nS, self.nS, self.nS).copy())
        delPi = (delPi.reshape(self.nS, self.nA, 1, 1, self.nS).copy())
        pi = (pi.reshape(self.nS, self.nA, 1, 1, 1).copy())
        return np.sum(P * (delPi * D + pi * delD), axis=(1, 2))

    def new_delQ(self, delPi: np.ndarray, pi: np.ndarray):
        # s_Q, a, s', a', s_r
        I = (self.I.reshape(self.nS, 1, 1, 1, self.nS))
        P = (self.P.reshape(self.nS, self.nA, self.nS, 1, 1))
        Q = (self.Q.reshape(1, 1, self.nS, self.nA, 1))
        delPi = (delPi.reshape(1, 1, self.nS, self.nA, self.nS))
        delQ = (self.delQ.reshape(1, 1, self.nS, self.nA, self.nS))
        pi = (pi.reshape(1, 1, self.nS, self.nA, 1))
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
