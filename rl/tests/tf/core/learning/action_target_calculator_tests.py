import logging
import unittest

import numpy as np
import scipy.stats
import tensorflow as tf
from mock import MagicMock

from rl.lib.timer import Timer
from rl.tf.core.learning.action_target_calculator import \
    QLearningActionTargetCalculator, \
    SarsaActionTargetCalculator, \
    ExpectedSarsaActionTargetCalculator

N = 10000

logging.basicConfig(level=logging.DEBUG)


class SarsaActionTargetCalculatorTests(unittest.TestCase):
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

        with Timer('Generating %i Samples' % N):
            samples = generate_samples(get_target, num_samples=N)

        itemfreq = scipy.stats.itemfreq(samples)
        itemfreq = dict(itemfreq)

        f_exp = N * a_probabilities
        f_obs = [itemfreq[0.0], itemfreq[1.0], itemfreq[2.0]]
        self.do_chi_squared_htest(f_obs, f_exp)

    def do_chi_squared_htest(self, f_obs, f_exp):
        chi = scipy.stats.chisquare(f_obs, f_exp)
        self.assertTrue(chi.pvalue > 0.01, chi.pvalue)


class QLearningActionTargetCalculatorTests(unittest.TestCase):

    def test_calculate(self):
        rl_system = MagicMock()
        calculator = QLearningActionTargetCalculator(
            rl_system=rl_system,
            discount_factor=1.0)

        reward = tf.Variable(1.0)
        a_next_state_action_values = np.array([1.0, 2.0, 3.0])
        t_next_state_action_values = tf.Variable(a_next_state_action_values, dtype=tf.float32)
        target = calculator.calculate(reward, t_next_state_action_values)

        actual = evaluate_tensor(target)

        self.assertEqual(4.0, actual)

    def test_vectorized_1d(self):

        rewards = tf.Variable([1.0, 2.0])
        next_state_action_values = tf.Variable(
            [
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        )

        calculator = QLearningActionTargetCalculator(
            rl_system=MagicMock(),
            discount_factor=1.0)
        t_targets = calculator.vectorized_1d(rewards, next_state_action_values)
        a_targets = evaluate_tensor(t_targets)

        expected = np.array([1.0 + 5.0, 2.0 + 6.0])

        np.testing.assert_array_equal(expected, a_targets)


class ExpectedSarsaActionTargetCalculatorTests(unittest.TestCase):
    def test_calculate(self):
        rl_system = MagicMock(num_actions=3)
        a_probabilities = np.array([0.5, 0.3, 0.2])
        t_probabilities = tf.Variable(a_probabilities, dtype=tf.float32)
        rl_system.policy.calculate_action_value_probabilities.return_value = t_probabilities

        calculator = ExpectedSarsaActionTargetCalculator(
            rl_system=rl_system,
            discount_factor=1.0
        )

        reward = tf.Variable(1.0)
        a_next_state_action_values = np.array([1.0, 2.0, 3.0])
        t_next_state_action_values = tf.Variable(a_next_state_action_values, dtype=tf.float32)
        target = calculator.calculate(reward, t_next_state_action_values)
        actual = evaluate_tensor(target)
        desired = 1.0 + np.dot(a_probabilities, a_next_state_action_values)

        np.testing.assert_almost_equal(desired, actual)


def generate_samples(t, num_samples=1000):
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
        return b_i + 1, b_result

    i, result = tf.while_loop(cond, body, (i, result))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        return sess.run(result.stack())


def evaluate_tensor(t):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        return sess.run(t)


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
