import logging
import unittest

import numpy as np
from mock import MagicMock

from rl.core.experience import Episode
from rl.core.learning.target_array_calculator import \
    build_q_learning_target_array_calculator, \
    build_sarsa_target_array_calculator, \
    VectorizedStateMachineTargetArrayCalculator
from rl.tests.core.learning.mock_rl_system import MockState1, MockState2A, MockState2B, MockState3
from rl.tests.core.learning.mock_rl_system import MockSystem
from rl.core.state import IntExtState, State

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


class VectorizedStateMachineTargetArrayCalculatorTest(unittest.TestCase):

    def setUp(self):

        action_target_calculator = MagicMock()
        rl_system = MagicMock(
            num_internal_states=2,
            num_actions=[2, 2]
        )

        self.calculator = VectorizedStateMachineTargetArrayCalculator(
            rl_system,
            action_target_calculator,
        )

    def test_get_state_targets(self):

        external_state = MagicMock()
        int_ext_state = IntExtState(0, external_state)

        targets = self.calculator.get_state_targets(int_ext_state)

        self.assertEqual(2, len(targets))

    def test_get_target_array(self):

        external_state1 = MagicMock()
        int_ext_state1 = IntExtState(0, external_state1)

        external_state2 = MagicMock()
        int_ext_state2 = IntExtState(1, external_state2)

        states = [int_ext_state1, int_ext_state2]

        targets = self.calculator.get_target_array(states)

        self.assertEqual((2, 4), targets.shape)


if __name__ == '__main__':
    unittest.main()