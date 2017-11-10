import logging
import unittest

import numpy as np
from unittest.mock import MagicMock

from rl.core.rl_system import RLSystem
from rl.core.policy import EquiProbableRandomPolicy, EpsilonGreedyPolicy, SoftmaxPolicy, Policy

logging.basicConfig(level=logging.DEBUG)


class RLSystemTests(unittest.TestCase):

    def test_choose_action(self):
        """
        Test that choose action returns a value in the expected range
        :return:
        """
        rl_system = RLSystem()
        rl_system.calculate_action_value_probabilities = MagicMock(return_value=np.array([0.5, 0.5]))

        num_true = 0
        n = 1000
        for _ in range(n):
            action = rl_system.choose_action(state=None)
            num_true += action

        # p=0.5, q=0.5, var = npq = 1000 * 0.5 * 0.5 = 250
        # sd = sqrt(250) = 15.8
        # Expect num_true to be within 4 standard deviations, which is 63
        # 500-63 = 437 <= num_true <= 500 + 63

        logging.info('Num True: %s' % num_true)
        self.assertTrue(num_true > 437, 'Failure would constitute a 4-sigma event - try again!')
        self.assertTrue(num_true < 563, 'Failure would constitute a 4-sigma event - try again!')


    def test_calculation_state_probabilities_equiprobable(self):
        """
        Test that choose action returns a value in the expected range
        :return:
        """

        rl_system = RLSystem(policy=EquiProbableRandomPolicy())
        rl_system.action_value_function = MagicMock(return_value=np.array([0.5, 0.5]))

        probabilities = rl_system.calculate_action_value_probabilities(state=None)
        expected = np.array([0.5, 0.5])
        np.testing.assert_array_equal(expected, probabilities)

    def test_calculate_state_probabilities_softmax(self):

        rl_system = RLSystem(policy=SoftmaxPolicy())
        rl_system.action_value_function = MagicMock(return_value=np.array([3.0, 1.0, 0.2]))

        result = rl_system.calculate_action_value_probabilities(state=None)
        # From https://stackoverflow.com/questions/34968722/softmax-function-python
        expected = [0.8360188, 0.11314284, 0.05083836]
        np.testing.assert_almost_equal(expected, result)


if __name__ == '__main__':
    unittest.main()
