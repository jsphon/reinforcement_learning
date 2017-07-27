"""
Test for Example 4.1 of Sutton
"""

import unittest
import logging

from rl.environments.grid_world import GridRewardFunction, GridWorld
from rl.environments.grid_world import GridState, GridActionValueFunction
from rl.core.policy import EpsilonGreedyPolicy
import numpy as np

#logging.basicConfig(level=logging.DEBUG)

N = 1000


class GridWorldTests(unittest.TestCase):
    def setUp(self):
        self.world = GridWorld()

    def test_get_value_grid(self):
        result = self.world.get_value_grid()
        self.assertEqual((4, 4), result.shape)
        logging.info(result)

        # elements (0, 0) and (3, 3) are the terminal states, so should have value np.nan
        msg = 'Element (0, 0) is %s' % str(result[0, 0])
        self.assertTrue(np.isnan(result[0, 0]), msg)
        self.assertTrue(np.isnan(result[3, 3]))

    def test_get_action_grid(self):
        self.world.policy.epsilon = 0.0
        result = self.world.get_action_grid()
        logging.info(result)
        self.assertEqual((4, 4), result.shape)

        self.assertEqual(-1, result[0, 0])
        self.assertEqual(-1, result[3, 3])

    def test_get_action_grid_string(self):
        self.world.policy.epsilon = 0.0
        result = self.world.get_action_grid_string()
        logging.info('\n' + result)

    def test_get_action_grid_string2(self):
        self.world.policy.epsilon = 0.0
        self.world.get_action_grid = lambda:np.array([
            [0, 1, 2, 3],
            [3, 2, 1, 0],
            [0, 0, 1, 1],
            [2, 2, 3, 3]
        ])

        result = self.world.get_action_grid_string()
        expected = r"""
^v<>
><v^
^^vv
<<>>
""".strip()
        self.assertEqual(expected, result)

    def test_get_reward_grid(self):
        result = self.world.get_reward_grid()

        expected = np.array(
            [
                [0, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, 0]
            ])

        np.testing.assert_array_equal(expected, result)


class GridActionValueFunctionTests(unittest.TestCase):
    def test___call__(self):
        state = GridState(player=(0, 0))
        f = GridActionValueFunction()
        result = f(state)
        self.assertEqual((4, ), result.shape)

    def test_on_list(self):
        states = [GridState(player=(0, 0)), GridState(player=(0, 1))]
        f = GridActionValueFunction()
        result = f.on_list(states)
        self.assertEqual((2, 4), result.shape)


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
                matches = [x for x in all_states if x.player == pos]
                self.assertEqual(1, len(matches))


if __name__ == '__main__':
    unittest.main()
