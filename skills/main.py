"""
the algorithm
"""
from gym.wrappers import TimeLimit

from skills.gridworld import GoalGridworld
from skills.trainer import Trainer
import numpy as np
import plotly
import plotly.graph_objs as go

if __name__ == '__main__':
    np.random.seed(0)

    desc = ['◻' * 5] * 5
    ENV = TimeLimit(
        max_episode_steps=10,
        env=GoalGridworld(
            desc=desc,
            rewards=dict(),
            terminal='T',
            start_states='◻',
        ))
    ENV.seed(0)

    # actions=np.array([[0, 1], [0, 0], [0, -1]]),
    # action_strings="▶s◀")
    def train(baseline: bool):
        return Trainer(env=ENV).train(baseline=baseline)

    e_x, e_y = zip(*enumerate(train(baseline=False)))
    b_x, b_y = zip(*enumerate(train(baseline=True)))
    plotly.offline.plot([
        go.Scatter(x=e_x, y=e_y, name='experiment'),
        go.Scatter(x=b_x, y=b_y, name='baseline')
    ],
                        auto_open=True)

    # _R, _Q, _D = optimize_reward(
    #     lambda s: float(s == 0),
    #     env=ENV,
    # )
    # print('_R')
    # print(_R)
    # print('_Q')
    # print(_Q)
    # print('_D')
    # print(_D)
