"""
the algorithm
"""
from gym.wrappers import TimeLimit

from skills.gridworld import GoalGridworld
from skills.model_free import Trainer

if __name__ == '__main__':
    ENV = TimeLimit(
        max_episode_steps=10,
        env=GoalGridworld(
            desc=[
                '◻◻◻◻◻',
                '◻◻◻◻◻',
                '◻◻◻◻◻',
                '◻◻◻◻◻',
                '◻◻◻◻S',
            ],
            rewards=dict(),
            terminal='T',
            start_states='S',
        ))
    # actions=np.array([[0, 1], [0, 0], [0, -1]]),
    # action_strings="▶s◀")
    Trainer(env=ENV, len_action_history=3).train()

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
