import unittest

import numpy as np
import tensorflow as tf
import scipy.stats

from rl.tf.core.policy import EpsilonGreedyPolicy
from rl.tf.core.system import System


class MyTestCase(unittest.TestCase):

    def test_choose_action(self):
        """
        Test that choose_action returns actions with the desired
        distribution

        """
        probabilities = [[.2, .3, .5]]
        a_logits = np.log(probabilities)
        t_logits = tf.Variable(a_logits)

        sys = System()
        sys.calculate_state_probabilities = lambda x: t_logits
        action = sys.choose_action(None)

        f_obs = np.array([0, 0, 0])
        N = 10000
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for _ in range(N):
                actual = sess.run(action)
                f_obs[actual] += 1

        f_exp = N * np.array(probabilities)
        result = scipy.stats.chisquare(f_obs, f_exp[0])
        self.assertGreater(result.pvalue, 0.01)

    def test_calculate_action_value_probabilities(self):

        test_data = (
            ((0.05, 0.05, 0.9), (1.0, 2.0, 3.0)),
            ((0.05, 0.9, 0.05), (1.0, 3.0, 2.0)),
            ((0.9, 0.05, 0.05), (3.0, 2.0, 1.0)),
        )

        for desired, action_values in test_data:

            a_action_values = np.array(action_values)
            t_action_values = tf.Variable(a_action_values, dtype=tf.float32)

            sys = System()
            sys.action_value_function = lambda x: t_action_values
            sys.policy = EpsilonGreedyPolicy(0.1)

            t_probabilities = sys.calculate_action_value_probabilities(state=None)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                a_probabilities = sess.run(t_probabilities)

            a_desired = np.array(desired)
            np.testing.assert_array_almost_equal(a_desired, a_probabilities)


if __name__ == '__main__':
    unittest.main()
