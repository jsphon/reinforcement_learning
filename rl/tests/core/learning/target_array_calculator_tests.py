import logging
import unittest

import numpy as np

from rl.core.experience import Episode
from rl.core.learning.target_array_calculator import \
    build_q_learning_target_array_calculator, \
    build_sarsa_target_array_calculator
from rl.tests.core.learning.mock_rl_system import MockState1, MockState2A, MockState2B, MockState3
from rl.tests.core.learning.mock_rl_system import MockSystem

N = 1000

logging.basicConfig(level=logging.DEBUG)


class QLearningTargetArrayCalculatorTests(unittest.TestCase):

    def test_q_learning_get_target_array(self):
        states = [MockState1(), MockState2A(), MockState3()]
        actions = [0, 0]
        rewards = [0, 0]
        episode = Episode(states=states, actions=actions, rewards=rewards)

        calculator = build_q_learning_target_array_calculator(MockSystem())
        targets = calculator.get_target_array(episode)

        # Get this from MockActionValueFunction
        expected = np.array([1, 0])
        np.testing.assert_array_almost_equal(expected, targets)

    def test_q_learning_get_target_array2(self):
        states = [MockState1(), MockState2B(), MockState3()]
        actions = [1, 0]
        rewards = [0, 10]
        episode = Episode(states=states, actions=actions, rewards=rewards)

        calculator = build_q_learning_target_array_calculator(MockSystem())
        targets = calculator.get_target_array(episode)

        expected = np.array([
            0.0 + 1,
            10.0 + 0.0
        ])

        np.testing.assert_almost_equal(expected, targets)


class SarsaTargetArrayCalculatorTests(unittest.TestCase):

    def setUp(self):
        mock_system = MockSystem()
        self.calculator = build_sarsa_target_array_calculator(mock_system, discount_factor=0.9)

    def test_get_target_array1(self):
        states = [MockState1(), MockState2A(), MockState3()]
        rewards = [11, 33]
        actions = [0, 0]
        episode = Episode(states, actions, rewards)

        targets = self.calculator.get_target_array(episode)

        expected = np.array([
            11 + 0.9 * 1.0,
            33 + 0.9 * 0.0
        ])

        np.testing.assert_almost_equal(expected, targets)

    def test_get_target_array2(self):
        states = [MockState1(), MockState2B(), MockState3()]
        rewards = [22, 33]
        actions = [1, 0]
        episode = Episode(states, actions, rewards)

        targets = self.calculator.get_target_array(episode)

        expected = np.array([
            22.0 + 0.9 * 0,
            33.0 + 0.9 * 0
        ])

        np.testing.assert_array_almost_equal(expected, targets)


    # Where did these come from?
    # def test_get_state_targets1(self):
    #
    #     targets = self.calculator.get_state_targets(state=MockState1(),
    #                                         action=0,
    #                                         reward=11)
    #
    #     # Expected0 = reward(action|state) + gamma * Q(next_state|next_action)
    #     #           = 11 + 0.9 * 1
    #     #           = 11.9
    #     # Expected1 = Q(MockState1(), action=1)
    #     #           = 2
    #     expected = np.array([11.9, 2.0])
    #     np.testing.assert_almost_equal(expected, targets)
    #
    # def test_get_state_targets2A(self):
    #
    #     targets = self.calculator.get_state_targets(state=MockState2A(),
    #                                         action=0,
    #                                         reward=1)
    #
    #     # Expected0 = reward(action|state) + gamma * Q(next_state|next_action)
    #     #           = 1 + 0.9 * 0
    #     #           = 1.0
    #     # Expected1 = Q(MockState2A(), action=1)
    #     #           = 0.0
    #     expected = np.array([1.0, 0.0])
    #     np.testing.assert_almost_equal(expected, targets)
    #
    # def test_get_state_targets2B(self):
    #
    #     targets = self.calculator.get_state_targets(state=MockState2B(),
    #                                         action=0,
    #                                         reward=2)
    #
    #     # Expected0 = reward(action|state) + gamma * Q(next_state|next_action)
    #     #           = 2 + 0.9 * 0
    #     #           = 2.0
    #     # Expected1 = Q(MockState2B(), action=1)
    #     #           = 1.0
    #     expected = np.array([2.0, 1.0])
    #     np.testing.assert_almost_equal(expected, targets)


if __name__ == '__main__':
    unittest.main()
