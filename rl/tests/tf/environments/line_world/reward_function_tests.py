import logging

from rl.tests.tf.utils import evaluate_tensor

logging.getLogger("tensorflow").setLevel(logging.WARNING)

import tensorflow as tf
import unittest
import numpy as np

from rl.tf.environments.line_world.constants import TARGET
from rl.tf.environments.line_world.reward_function import RewardFunction
from rl.tf.environments.line_world.model import LineWorldModel


class MyTestCase(tf.test.TestCase):

    def test_state_rewards(self):
        state = tf.Variable(TARGET + 1)
        next_states = tf.Variable([TARGET, TARGET + 2])

        rf = RewardFunction()

        t_rewards = rf.state_rewards(state, next_states)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            result = t_rewards.eval()

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

            with self.test_session() as sess:
                sess.run(tf.global_variables_initializer())
                a_rewards = t_rewards.eval()
            expected_rewards = np.array(expected_rewards)
            np.testing.assert_array_equal(expected_rewards, a_rewards)

    def test_state_rewards_vectorized(self):

        positions = np.arange(10)
        t_states = tf.Variable(positions)

        a_next_states = np.array([
            [0, 0, 1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 9]
        ])
        t_next_states = tf.Variable(a_next_states)

        rf = RewardFunction()
        result = rf.state_rewards_vectorized(t_states, t_next_states)

        expected = np.array([
            [-1, -1, -1, 10, -1, -1, -1, -1, -1, -1],
            [-1, 10, -1, -1, -1, -1, -1, -1, -1, -1],
        ])

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            a_result = result.eval()

        self.assertAllClose(expected, a_result)


if __name__ == '__main__':
    unittest.main()
