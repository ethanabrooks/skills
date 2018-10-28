"""
the algorithm
"""
# stdlib
import argparse

# third party
from gym.wrappers import TimeLimit
import numpy as np
import plotly
import plotly.graph_objs as go
import random

# first party
from skills.gridworld import GoalGridworld
from skills.trainer import Trainer


def main(iterations: int, slack: int):
    # desc = [
    # '___#___',
    # '_______',
    # '___#___',
    # '#_###_#',
    # '___#___',
    # '_______',
    # '___#___',
    # ]
    desc = [
        '__________',
    ]
    ENV = TimeLimit(
        max_episode_steps=30,
        env=GoalGridworld(
            desc=desc,
            actions=np.array([[0, 1], [-1, 0]]),
            action_strings="▶◀",
            rewards=dict(),
            terminal='T',
            start_states='_',
            blocked_states='#',
        ))

    def seed(n):
        ENV.seed(n)
        np.random.seed(n)
        random.seed(n)

    # actions=np.array([[0, 1], [0, 0], [0, -1]]),
    # action_strings="▶s◀")
    def train(baseline: bool):
        return Trainer(
            env=ENV, slack_factor=slack).train(
                iterations=iterations, baseline=baseline)

    print('baseline')
    seed(0)
    b_x, b_y = zip(*enumerate(train(baseline=True)))
    print('experiment')
    seed(0)
    e_x, e_y = zip(*enumerate(train(baseline=False)))
    fig = go.Figure(
        data=[
            go.Scatter(x=e_x, y=e_y, name='experiment'),
            go.Scatter(x=b_x, y=b_y, name='baseline')
        ],
        layout=dict(yaxis=dict(type='log')))
    plotly.offline.plot(fig, auto_open=True)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iterations', type=int, required=True)
    parser.add_argument('-s', '--slack', type=int, required=True)
    main(**vars(parser.parse_args()))


if __name__ == '__main__':
    cli()
