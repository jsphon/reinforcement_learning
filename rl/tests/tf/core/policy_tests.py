import unittest

import numpy as np
import tensorflow as tf

from rl.tf.core.policy import EpsilonGreedyPolicy, SoftmaxPolicy


class SoftmaxPolicyTests(unittest.TestCase):

    def test_calculate_action_value_probabilities(self):

        test_data = (
            [0.1, 0.2, 0.3],
            [0.1, 0.3, 0.2],
            [0.3, 0.2, 0.1],
        )

        for action_values in test_data:
            t_action_values = tf.Variable(action_values)
            policy = SoftmaxPolicy()

            probabilities = policy.calculate_action_value_probabilities(t_action_values)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                actual = sess.run(probabilities)

            desired = softmax(action_values)
            np.testing.assert_almost_equal(actual, desired)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class EpsilonGreedyPolicyTests(unittest.TestCase):

    def test_calculate_action_value_probabilities(self):

        test_data = (
            [[0.05, 0.05, 0.9], [0.1, 0.2, 0.3]],
            [[0.05, 0.9, 0.05], [0.1, 0.3, 0.2]],
            [[0.9,  0.05, 0.05], [0.3, 0.2, 0.1]]
        )

        for expected, action_values in test_data:
            action_values = tf.Variable(action_values)
            epsilon = tf.Variable(0.1)
            policy = EpsilonGreedyPolicy(epsilon)

            probabilities = policy.calculate_action_value_probabilities(action_values)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                result = sess.run(probabilities)

            np.testing.assert_almost_equal(expected, result)


if __name__ == '__main__':
    unittest.main()
