import unittest
import tensorflow as tf

import numpy as np
from rl.tf.policy import EpsilonGreedyPolicy


class MyTestCase(unittest.TestCase):

    def test_something(self):
        self.assertEqual(True, False)

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
