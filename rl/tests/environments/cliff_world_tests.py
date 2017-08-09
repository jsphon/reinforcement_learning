
import unittest
import logging

from rl.environments.cliff_world import CliffWorld, walked_off_cliff, GridState
import numpy as np


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



if __name__ == '__main__':
    unittest.main()
