import unittest

import numpy as np
import tensorflow as tf
import scipy.stats

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


if __name__ == '__main__':
    unittest.main()
