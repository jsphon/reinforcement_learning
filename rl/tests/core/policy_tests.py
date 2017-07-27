"""
Test for Example 4.1 of Sutton
"""

import logging
import unittest

import numpy as np
from unittest.mock import MagicMock

from rl.core.policy import EquiProbableRandomPolicy, EpsilonGreedyPolicy, SoftmaxPolicy, Policy

logging.basicConfig(level=logging.DEBUG)


class PolicyTests(unittest.TestCase):

    def test_choose_action(self):
        """
        Test that choose action returns a value in the expected range
        :return:
        """
        policy = Policy(rl_system=MagicMock(num_actions=2))
        policy.calculate_state_probabilities = MagicMock(return_value=np.array([0.5, 0.5]))

        num_true = 0
        n = 1000
        for _ in range(n):
            action = policy.choose_action(state=None)
            num_true += action

        # p=0.5, q=0.5, var = npq = 1000 * 0.5 * 0.5 = 250
        # sd = sqrt(250) = 15.8
        # Expect num_true to be within 4 standard deviations, which is 63
        # 500-63 = 437 <= num_true <= 500 + 63

        logging.info('Num True: %s' % num_true)
        self.assertTrue(num_true > 437, 'Failure would constitute a 4-sigma event - try again!')
        self.assertTrue(num_true < 563, 'Failure would constitute a 4-sigma event - try again!')


class EquiprobableRandomPolicyTests(unittest.TestCase):

    def test_choose_action(self):
        rl_system = MagicMock()
        rl_system.num_actions = 2

        policy = EquiProbableRandomPolicy(rl_system)

        num_true = 0
        n = 1000
        for _ in range(n):
            action = policy.choose_action(state=None)
            num_true += action

        # p=0.5, q=0.5, var = npq = 1000 * 0.5 * 0.5 = 250
        # sd = sqrt(250) = 15.8
        # Expect num_true to be within 4 standard deviations, which is 63
        # 500-63 = 437 <= num_true <= 500 + 63

        print('Num True: %s' % num_true)
        logging.info('Num True: %s' % num_true)
        self.assertTrue(num_true>437, 'Failure would constitute a 4-sigma event - try again!')
        self.assertTrue(num_true<563, 'Failure would constitute a 4-sigma event - try again!')

    def test_calculate_probabilities(self):

        rl_system = MagicMock()
        rl_system.num_actions = 4
        policy = EquiProbableRandomPolicy(rl_system)
        result = policy.calculate_state_probabilities(state=None)

        expected = np.array([0.25, 0.25, 0.25, 0.25])
        np.testing.assert_array_equal(expected, result)


class EpsilonGreedyPolicyTests(unittest.TestCase):

    def test_calculate_probabilities(self):

        _state = MagicMock()

        def action_value_function(state):
            if state == _state:
                return np.array([0.1, 0.2, 0.3])
            else:
                raise ValueError()

        rl_system = MagicMock(action_value_function=action_value_function)
        rl_system.num_actions = 3

        policy = EpsilonGreedyPolicy(rl_system, epsilon=0.1)

        # 90% chance of choosing the third action
        # 5% chance of choosing the first or second action
        result = policy.calculate_state_probabilities(state=_state)
        expected = np.array([0.05, 0.05, 0.9])

        np.testing.assert_almost_equal(expected, result)


class SoftmaxPolicyTests(unittest.TestCase):

    def test_calculate_probabilities(self):
        """
        Expected vaues from https://stackoverflow.com/questions/34968722/softmax-function-python
        :return:
        """

        _state = MagicMock()

        def action_value_function(state):
            if state == _state:
                return np.array([3.0, 1.0, 0.2])
            else:
                raise ValueError()

        rl_system = MagicMock(action_value_function=action_value_function)
        rl_system.num_actions = 3

        policy = SoftmaxPolicy(rl_system)
        result = policy.calculate_state_probabilities(_state)
        # From https://stackoverflow.com/questions/34968722/softmax-function-python
        expected = [0.8360188, 0.11314284, 0.05083836]
        np.testing.assert_almost_equal(expected, result)

    def test_calculate_probabilities_grid_world(self):
        from rl.environments.grid_world import GridWorld, GridState

        state = GridState(player=(1, 1))
        grid_world = GridWorld()
        p = grid_world.policy.calculate_state_probabilities(state)
        logging.info('GridWorld Probabilities: %s' % str(p))

        grid_world.policy = SoftmaxPolicy(grid_world)
        p = grid_world.policy.calculate_state_probabilities(state)
        action = grid_world.policy.choose_action(state)


if __name__ == '__main__':
    unittest.main()
