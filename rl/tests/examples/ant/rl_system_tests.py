import unittest

import numpy as np
from rl.examples.ant.rl_system import AntWorld, calculate_greedy_actions
from rl.examples.ant.state import AntState
from rl.core.state import IntExtState
from rl.core.learning.learner import build_learner
from rl.core.state import StateList


class MyTestCase(unittest.TestCase):

    def test_something(self):

        world = AntWorld()

        learner = build_learner(world, calculator_type='modelbasedstatemachine')

        states_list = StateList([AntState(position=position) for position in range(10)])

        for _ in range(1000):
            learner.learn(states_list, epochs=1)

        greedy_actions = calculate_greedy_actions(world)

        print(greedy_actions)
        expected_greedy_actions = r"""
11x0000000
11111111x0
""".strip()
        self.assertEqual(expected_greedy_actions, greedy_actions)

        action_values = world.calculate_action_values()
        print(action_values)

        expected_action_values = [
            np.array([
                [8, 9],
                [8, 10],
                [np.nan, np.nan],
                [10, 8],
                [9, 7],
                [8, 6],
                [7, 5],
                [6, 5],
                [5, 4],
                [4, 3]
            ]),
            np.array([
                [6, 8],
                [7, 9],
                [8, 10],
                [9, 11],
                [10, 12],
                [11, 13],
                [12, 14],
                [13, 15],
                [np.nan, np.nan],
                [15, 14]
            ]),
        ]

        e0 = expected_action_values[0]
        r0 = action_values[0]
        r0[np.isnan(e0)] = np.nan
        np.testing.assert_array_almost_equal(e0, r0, decimal=0)

        e1 = expected_action_values[1]
        r1 = action_values[1]
        r1[np.isnan(e1)] = np.nan
        np.testing.assert_array_almost_equal(e1, r1, decimal=0)


        # Why is this [10, 10] ?


if __name__ == '__main__':
    unittest.main()
