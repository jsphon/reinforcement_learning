import logging

from rl.tests.tf.utils import evaluate_tensor

logging.getLogger("tensorflow").setLevel(logging.WARNING)

import tensorflow as tf
import unittest
import numpy as np

from rl.tf.environments.line_world.constants import TARGET
from rl.tf.environments.line_world.reward_function import RewardFunction
from rl.tf.environments.line_world.model import LineWorldModel


class MyTestCase(unittest.TestCase):

    def test_state_rewards(self):
        state = tf.Variable(TARGET + 1)
        next_states = tf.Variable([TARGET, TARGET + 2])

        rf = RewardFunction()

        t_rewards = rf.state_rewards(state, next_states)

        result = evaluate_tensor(t_rewards)

        expected = np.array([10, -1])
        np.testing.assert_array_equal(expected, result)

    def test_state_rewards2(self):

        model = LineWorldModel()
        rf = RewardFunction()

        data = (
            (0, [-1., -1.]),
            (1, [-1., 10.]),
            (2, [-1., -1.]),
            (3, [10., -1.]),
            (4, [-1., -1.]),
            (5, [-1., -1.]),
            (6, [-1., -1.]),
            (7, [-1., -1.]),
            (8, [-1., -1.]),
            (9, [-1., -1.]),
        )
        for position, expected_rewards in data:

            t_state = tf.Variable(position)
            t_next_states = model.apply_actions(t_state)
            t_rewards = rf.state_rewards(t_state, t_next_states)
            print(evaluate_tensor(t_rewards))

            a_rewards = evaluate_tensor(t_rewards)
            expected_rewards = np.array(expected_rewards)
            np.testing.assert_array_equal(expected_rewards, a_rewards)


if __name__ == '__main__':
    unittest.main()
