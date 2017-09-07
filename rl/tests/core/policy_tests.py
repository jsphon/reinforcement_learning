"""
Test for Example 4.1 of Sutton
"""

import logging
import unittest

import numpy as np
from unittest.mock import MagicMock

from rl.core.policy import EquiProbableRandomPolicy, EpsilonGreedyPolicy, SoftmaxPolicy, Policy

logging.basicConfig(level=logging.DEBUG)


class EquiprobableRandomPolicyTests(unittest.TestCase):

    def test_calculate_action_value_probabilities(self):

        policy = EquiProbableRandomPolicy()

        probabilities = policy.calculate_action_value_probabilities([1, 1])

        expected = np.array([0.5, 0.5])
        np.testing.assert_array_equal(expected, probabilities)


class EpsilonGreedyPolicyTests(unittest.TestCase):

    def test_calculate_probabilities(self):

        policy = EpsilonGreedyPolicy(epsilon=0.1)

        # 90% chance of choosing the third action
        # 5% chance of choosing the first or second action
        result = policy.calculate_action_value_probabilities(np.array([0.1, 0.2, 0.3]))
        expected = np.array([0.05, 0.05, 0.9])

        np.testing.assert_almost_equal(expected, result)


class SoftmaxPolicyTests(unittest.TestCase):

    def test_calculate_probabilities(self):
        """
        Expected vaues from https://stackoverflow.com/questions/34968722/softmax-function-python
        :return:
        """

        policy = SoftmaxPolicy()
        result = policy.calculate_action_value_probabilities(np.array([3.0, 1.0, 0.2]))
        # From https://stackoverflow.com/questions/34968722/softmax-function-python
        expected = [0.8360188, 0.11314284, 0.05083836]
        np.testing.assert_almost_equal(expected, result)


if __name__ == '__main__':
    unittest.main()
