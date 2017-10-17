
import logging
import textwrap
import unittest

import numpy as np

from rl.core.learning.learner import build_learner
from rl.environments.cliff_world import CliffWorld, GridState
from rl.environments.cliff_world import walked_off_cliff
from rl.lib.timer import Timer


class CliffWorldTests(unittest.TestCase):

    def setUp(self):
        self.world = CliffWorld()

    def test_is_terminal_grid(self):
        result = self.world.is_terminal_grid()
        self.assertEqual((4, 12), result.shape)
        logging.info(result)

        expected = np.array([
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=np.bool)
        np.testing.assert_array_equal(expected, result)


class ModuleTests(unittest.TestCase):

    def test_walked_off_cliff(self):

        self.assertTrue(walked_off_cliff(GridState((0, 1))))


class QLearningIntegrationTests(unittest.TestCase):

    def test_training(self):
        grid_world = CliffWorld()
        grid_world.action_value_function.learning_rate = 0.05

        states = GridState.all()

        learner = build_learner(grid_world, learning_algo='qlearning')

        with Timer('training') as t:
            for i in range(500):
                learner.learn(states)

        value_grid = grid_world.get_value_grid()
        nan = np.nan
        expected = np.array([
             [-13.,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan],
             [-12., -11., -10.,  -9.,  -8.,  -7.,  -6.,  -5.,  -4.,  -3.,  -2.,  -1.],
             [-13., -12., -11., -10.,  -9.,  -8.,  -7.,  -6.,  -5.,  -4.,  -3.,  -2.],
             [-14., -13., -12., -11., -10.,  -9.,  -8.,  -7.,  -6.,  -5.,  -4.,  -3.],
        ]
        )
        np.testing.assert_array_almost_equal(expected, value_grid, decimal=1)


if __name__ == '__main__':
    unittest.main()
