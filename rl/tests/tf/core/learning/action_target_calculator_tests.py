import logging
import unittest
from mock import MagicMock
from collections import defaultdict

import numpy as np
import tensorflow as tf
from rl.lib.timer import Timer

import scipy.stats
from rl.tf.core.learning.action_target_calculator import \
    QLearningActionTargetCalculator,\
    SarsaActionTargetCalculator,\
    ExpectedSarsaActionTargetCalculator

N = 10000

logging.basicConfig(level=logging.DEBUG)


class SarsaActionTargetCalculatorTests(unittest.TestCase):

    # def test_calculate(self):
    #     """
    #     Test that calculate_action_target returns targets with the correct distributions
    #
    #     """
    #     rl_system = MagicMock()
    #     calculator = SarsaActionTargetCalculator(
    #         rl_system=rl_system,
    #         discount_factor=tf.Variable(1.0))
    #
    #     a_probabilities = np.array([0.5, 0.3, 0.2])
    #     t_probabilities = tf.Variable(a_probabilities)
    #     rl_system.policy.calculate_action_value_probabilities.return_value = t_probabilities
    #     action_values = tf.Variable([0.0, 1.0, 2.0])
    #     reward = tf.Variable(0.0)
    #     t_target = calculator.calculate(reward, action_values)
    #
    #     counts = defaultdict(float)
    #     with tf.Session() as sess:
    #         sess.run(tf.global_variables_initializer())
    #         for _ in range(N):
    #             result = sess.run(t_target)
    #             logging.info(result)
    #             counts[result] += 1
    #
    #     f_exp = N * a_probabilities
    #     f_obs = [counts[0.0], counts[1.0], counts[2.0]]
    #     print(counts)
    #     self.do_chi_squared_htest(f_obs, f_exp)

    def do_chi_squared_htest(self, f_obs, f_exp):
        chi = scipy.stats.chisquare(f_obs, f_exp)
        self.assertTrue(chi.pvalue > 0.01, chi.pvalue)

    def test_calculate2(self):
        """
        Test that calculate_action_target returns targets with the correct distributions

        """
        rl_system = MagicMock()
        calculator = SarsaActionTargetCalculator(
            rl_system=rl_system,
            discount_factor=tf.Variable(1.0))

        a_probabilities = np.array([0.5, 0.3, 0.2])
        t_probabilities = tf.Variable(a_probabilities)
        rl_system.policy.calculate_action_value_probabilities.return_value = t_probabilities
        action_values = tf.Variable([0.0, 1.0, 2.0])
        reward = tf.Variable(0.0)

        def get_target():
            return calculator.calculate(reward, action_values)

        with Timer('Generating Samples'):
            samples = self.generate_samples(get_target, num_samples=N)

        itemfreq = scipy.stats.itemfreq(samples)
        itemfreq = dict(itemfreq)

        f_exp = N * a_probabilities
        f_obs = [itemfreq[0.0], itemfreq[1.0], itemfreq[2.0]]
        self.do_chi_squared_htest(f_obs, f_exp)

    def generate_samples(self, t, num_samples=1000):
        """
        Generate samples from the tensor t
        Args:
            t:
            num_samples:

        Returns:

        """

        i = tf.Variable(0)
        result = tf.TensorArray(t().dtype, num_samples)

        def cond(b_i, result):
            return tf.less(b_i, num_samples)

        def body(b_i, b_result):
            b_result = b_result.write(b_i, t())
            return b_i+1, b_result

        i, result = tf.while_loop(cond, body, (i, result))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run(result.stack())


#
#
# class QLearningActionTargetCalculatorTests(unittest.TestCase):
#     def test_calculate(self):
#         rl_system = MagicMock()
#         calculator = QLearningActionTargetCalculator(
#             rl_system=rl_system,
#             discount_factor=1.0)
#
#         reward = 1.0
#         next_state_action_values = np.array([1.0, 2.0, 3.0])
#         result = calculator.calculate(reward, next_state_action_values)
#
#         self.assertEqual(4.0, result)
#
#
# class ExpectedSarsaActionTargetCalculatorTests(unittest.TestCase):
#
#     def test_calculate(self):
#         rl_system = MagicMock(num_actions=3)
#         probabilities = np.array([0.5, 0.3, 0.2])
#         rl_system.policy.calculate_action_value_probabilities.return_value = probabilities
#
#         calculator = ExpectedSarsaActionTargetCalculator(
#             rl_system=rl_system,
#             discount_factor=1.0)
#
#         reward = 1.0
#         next_state_action_values = np.array([1.0, 2.0, 3.0])
#         result = calculator.calculate(reward, next_state_action_values)
#         expected = 1.0 + np.dot(probabilities, next_state_action_values)
#
#         self.assertEqual(expected, result)



if __name__ == '__main__':
    unittest.main()
