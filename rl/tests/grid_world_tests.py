"""
Test for Example 4.1 of Sutton
"""


import unittest

from rl.grid_world import GridState
from rl.grid_world import GridRewardFunction, GridWorld, TabularValueFunction
from rl.policy import EquiProbableRandomPolicy
from rl.value import ValueFunction

N = 1000


class GridWorldTests(unittest.TestCase):

    def test___init__(self):
        world = GridWorld()

    def test_evaluate_policy(self):
        world = GridWorld()
        world.policy = EquiProbableRandomPolicy(4)
        result = world.evaluate_policy()

        self.assertIsInstance(result, ValueFunction)


class TransitionProbabilityTests(unittest.TestCase):

    def test_xxx(self):
        pass


class TabularValueFunctionTests(unittest.TestCase):

    def test_xxx(self):

        policy = EquiProbableRandomPolicy(num_actions=4)
        tvf = TabularValueFunction(policy)
        state = GridState((0, 0))

        result = tvf(state)

        self.assertEqual(0, result)


class GridStateTests(unittest.TestCase):

    def test_reward_function(self):

        for position in ((0, 0), (3, 3)):

            state = GridState(position)
            reward_function = GridRewardFunction()

            result = reward_function(None, None, state)

            self.assertEqual(0, result)

    def test_reward_function_non_terminal(self):

        all_states = [(i, j) for i in range(3) for j in range(4)]
        non_terminal_states = [x for x in all_states if x not in ((0, 0), (3, 3))]
        for position in non_terminal_states:
            state = GridState(position)
            reward_function = GridRewardFunction()

            result = reward_function(None, None, state)

            self.assertEqual(-1, result)

    def test_all(self):
        all_states = GridState.all()
        for i in range(4):
            for j in range(4):
                pos = (i, j)
                matches = [x for x in all_states if x.player==pos]
                self.assertEqual(1, len(matches))

if __name__ == '__main__':
    unittest.main()
