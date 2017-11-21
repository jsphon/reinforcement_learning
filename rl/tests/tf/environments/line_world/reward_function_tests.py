import logging

from rl.tests.tf.utils import evaluate_tensor

logging.getLogger("tensorflow").setLevel(logging.WARNING)

import tensorflow as tf
import unittest
import numpy as np

from rl.tf.environments.line_world.constants import TARGET
from rl.tf.environments.line_world.reward_function import RewardFunction


class MyTestCase(unittest.TestCase):
    def test_state_rewards(self):
        state = tf.Variable(TARGET + 1)
        next_states = tf.Variable([TARGET, TARGET + 2])

        rf = RewardFunction()

        t_rewards = rf.state_rewards(state, next_states)

        result = evaluate_tensor(t_rewards)

        expected = np.array([10, -1])
        np.testing.assert_array_equal(expected, result)


if __name__ == '__main__':
    unittest.main()
