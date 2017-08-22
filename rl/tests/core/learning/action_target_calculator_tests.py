import logging
import unittest
from mock import MagicMock
from collections import defaultdict

import numpy as np

import scipy.stats
from rl.core.learning.action_target_calculator import \
    QLearning,\
    SarsaActionTargetCalculator,\
    ExpectedSarsaActionTargetCalculator

N = 1000

logging.basicConfig(level=logging.DEBUG)


class SarsaActionTargetCalculatorTests(unittest.TestCase):

    def test_calculate(self):
        """
        Test that calculate_action_target returns targets with the correct distributions

        """
        rl_system = MagicMock()
        calculator = SarsaActionTargetCalculator(
            rl_system=rl_system,
            discount_factor=1.0)

        probabilities = np.array([0.5, 0.3, 0.2])
        rl_system.policy.calculate_action_value_probabilities.return_value = probabilities

        counts = defaultdict(float)
        for _ in range(N):
            result = calculator.calculate(0, [0, 1, 2])
            logging.info(result)
            counts[result] += 1

        f_exp = N * probabilities
        f_obs = [counts[0.0], counts[1.0], counts[2.0]]
        self.do_chi_squared_htest(f_obs, f_exp)

    def do_chi_squared_htest(self, f_obs, f_exp):
        chi = scipy.stats.chisquare(f_obs, f_exp)
        self.assertTrue(chi.pvalue > 0.01)


class QLearningActionTargetCalculatorTests(unittest.TestCase):
    def test_calculate(self):
        rl_system = MagicMock()
        calculator = QLearning(
            rl_system=rl_system,
            discount_factor=1.0)

        reward = 1.0
        next_state_action_values = np.array([1.0, 2.0, 3.0])
        result = calculator.calculate(reward, next_state_action_values)

        self.assertEqual(4.0, result)


class ExpectedSarsaActionTargetCalculatorTests(unittest.TestCase):

    def test_calculate(self):
        rl_system = MagicMock(num_actions=3)
        probabilities = np.array([0.5, 0.3, 0.2])
        rl_system.policy.calculate_action_value_probabilities.return_value = probabilities

        calculator = ExpectedSarsaActionTargetCalculator(
            rl_system=rl_system,
            discount_factor=1.0)

        reward = 1.0
        next_state_action_values = np.array([1.0, 2.0, 3.0])
        result = calculator.calculate(reward, next_state_action_values)
        expected = 1.0 + np.dot(probabilities, next_state_action_values)

        self.assertEqual(expected, result)



if __name__ == '__main__':
    unittest.main()
