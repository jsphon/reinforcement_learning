"""
Test for Example 4.1 of Sutton
"""

import unittest
import logging

from rl.environments.simple_grid_world import SimpleGridWorldRewardFunction, SimpleGridWorld
from rl.environments.simple_grid_world import SimpleGridState, GridActionValueFunction
import numpy as np

# logging.basicConfig(level=logging.DEBUG)

N = 1000


class GridWorldTests(unittest.TestCase):
    def setUp(self):
        self.world = SimpleGridWorld()

    def test_get_value_grid(self):
        result = self.world.get_value_grid()
        self.assertEqual((4, 4), result.shape)
        logging.info(result)

        # elements (0, 0) and (3, 3) are the terminal states, so should have value np.nan
        msg = 'Element (0, 0) is %s' % str(result[0, 0])
        self.assertTrue(np.isnan(result[0, 0]), msg)
        self.assertTrue(np.isnan(result[3, 3]))

    def test_get_greedy_action_grid(self):
        result = self.world.get_greedy_action_grid()
        logging.info(result)
        self.assertEqual((4, 4), result.shape)

        self.assertEqual(-1, result[0, 0])
        self.assertEqual(-1, result[3, 3])

    def test_get_greedy_action_grid_string(self):
        result = self.world.get_greedy_action_grid_string()
        logging.info('\n' + result)

    def test_get_greedy_action_grid_string2(self):
        self.world.get_greedy_action_grid = lambda: np.array([
            [0, 1, 2, 3],
            [3, 2, 1, 0],
            [0, 0, 1, 1],
            [2, 2, 3, 3]
        ])

        result = self.world.get_greedy_action_grid_string()
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
        state = SimpleGridState(player=(0, 0))
        f = GridActionValueFunction(shape=(4, 4), num_actions=4)
        result = f(state)
        self.assertEqual((4,), result.shape)

    def test_on_list(self):
        states = [SimpleGridState(player=(0, 0)), SimpleGridState(player=(0, 1))]
        f = GridActionValueFunction(shape=(4, 4), num_actions=4)
        result = f.on_list(states)
        self.assertEqual((2, 4), result.shape)


class GridStateTests(unittest.TestCase):
    def test_reward_function(self):

        for position in ((0, 0), (3, 3)):
            state = SimpleGridState(position)
            reward_function = SimpleGridWorldRewardFunction()

            result = reward_function(None, None, state)

            self.assertEqual(0, result)

    def test_reward_function_non_terminal(self):

        all_states = [(i, j) for i in range(3) for j in range(4)]
        non_terminal_states = [x for x in all_states if x not in ((0, 0), (3, 3))]
        for position in non_terminal_states:
            state = SimpleGridState(position)
            reward_function = SimpleGridWorldRewardFunction()

            result = reward_function(None, None, state)

            self.assertEqual(-1, result)


if __name__ == '__main__':
    unittest.main()
