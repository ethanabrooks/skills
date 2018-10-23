"""
the algorithm
"""
from skill.gridworld import Gridworld
from skill.model_based import RewardOptimizer

if __name__ == '__main__':
    ENV = Gridworld(
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
    )
    # actions=np.array([[0, 1], [0, 0], [0, -1]]),
    # action_strings="▶s◀")
    RewardOptimizer(visit_func=lambda s: float(s == 0),
                    env=ENV).train()

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
