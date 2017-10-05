import logging
import unittest

import numpy as np
from mock import MagicMock

from rl.core.experience import Episode
from rl.core.learning.target_array_calculator import build_target_array_calculator,\
    ModelBasedTargetArrayCalculator

from rl.tests.core.learning.mock_rl_system import MockState1, MockState2A, MockState2B, MockState3
from rl.tests.core.learning.mock_rl_system import MockSystem
from rl.core.state import IntExtState, State, StateList


N = 1000

logging.basicConfig(level=logging.DEBUG)


class ModelBasedTargetArrayCalculatorTests(unittest.TestCase):

    def setUp(self):
        rl_system = MockSystem()
        self.calculator = build_target_array_calculator(
            rl_system=rl_system,
            calculator_type='modelbased',
            learning_algo='qlearning'
        )

    def test_setUp(self):
        self.assertIsInstance(self.calculator, ModelBasedTargetArrayCalculator)

    def test_get_target_matrix(self):
        s1 = MockState1()
        s2a = MockState2A()
        s2b = MockState2B()
        s3 = MockState3()

        states = StateList([s1, s2a, s2b, s3])

        result = self.calculator.get_target_matrix(states)
        self.assertEqual((4, 2), result.shape)

        expected = np.array([[12, 23], [33, 33], [33, 33], [0, 0]])
        np.testing.assert_array_equal(expected, result)

    def test_get_state_targets_s1(self):
        s1 = MockState1()
        result = self.calculator.get_state_targets(s1)

        self.assertEqual((2, ), result.shape)
        self.assertEqual(12, result[0])
        self.assertEqual(23, result[1])

    def test_get_state_targets_s2a(self):
        s2a = MockState2A()
        result = self.calculator.get_state_targets(s2a)

        self.assertEqual((2, ), result.shape)
        self.assertEqual(33, result[0])
        self.assertEqual(33, result[1])

    def test_get_state_targets_s2b(self):
        s2b = MockState2B()
        result = self.calculator.get_state_targets(s2b)

        self.assertEqual((2, ), result.shape)
        self.assertEqual(33, result[0])
        self.assertEqual(33, result[1])

    def test_get_state_targets_s3(self):
        s3 = MockState3()
        result = self.calculator.get_state_targets(s3)

        self.assertEqual((2, ), result.shape)
        self.assertEqual(0, result[0])
        self.assertEqual(0, result[1])

    def test_get_state_action_target_s1(self):
        result = self.calculator.get_state_action_target(MockState1(), 0)
        self.assertEqual(12, result)

        result = self.calculator.get_state_action_target(MockState1(), 1)
        self.assertEqual(23, result)

    def test_get_state_action_target_s3(self):
        result = self.calculator.get_state_action_target(MockState3(), 0)
        self.assertEqual(0, result)

        result = self.calculator.get_state_action_target(MockState3(), 1)
        self.assertEqual(0, result)


class StateMachineTargetArrayCalculatorTests(unittest.TestCase):

    def setUp(self):
        rl_system = MagicMock(num_internal_states=2)
        self.calculator = build_target_array_calculator(
            rl_system=rl_system,
            calculator_type='modelbasedstatemachine',
            learning_algo='qlearning'
        )

    # def test_get_state_targets1(self):
    #
    #     internal_state = 0
    #     external_state = MagicMock(num_actions=2)
    #     int_ext_state = IntExtState(internal_state, external_state)
    #
    #     targets = self.calculator.get_state_targets(int_ext_state)
    #     self.assertEqual(2, targets.shape[0])
    #
    # def test_get_state_targets2(self):
    #
    #     internal_state = 1
    #     external_state = MagicMock()
    #     int_ext_state = IntExtState(internal_state, external_state)
    #
    #     targets = self.calculator.get_state_targets(int_ext_state)
    #     self.assertEqual(3, targets.shape[0])

    def test_get_target_matrix(self):

        external_state1 = MagicMock(num_actions=2)
        external_state2 = MagicMock(num_actions=2)
        external_states = [external_state1, external_state2]
        lst = StateList(external_states)
        targets = self.calculator.get_target_matrix(lst)
        print(targets)
        self.assertEqual(2, len(targets))
        #self.fail()

#
# class QLearningTargetArrayCalculatorTests(unittest.TestCase):
#
#     def test_q_learning_get_target_array(self):
#         states = [MockState1(), MockState2A(), MockState3()]
#         actions = [0, 0]
#         rewards = [0, 0]
#         episode = Episode(states=states, actions=actions, rewards=rewards)
#
#         calculator = build_q_learning_target_array_calculator(MockSystem())
#         targets = calculator.get_target_matrix(episode)
#
#         # Get this from MockActionValueFunction
#         expected = np.array([1, 0])
#         np.testing.assert_array_almost_equal(expected, targets)
#
#     def test_q_learning_get_target_array2(self):
#         states = [MockState1(), MockState2B(), MockState3()]
#         actions = [1, 0]
#         rewards = [0, 10]
#         episode = Episode(states=states, actions=actions, rewards=rewards)
#
#         calculator = build_q_learning_target_array_calculator(MockSystem())
#         targets = calculator.get_target_matrix(episode)
#
#         expected = np.array([
#             0.0 + 1,
#             10.0 + 0.0
#         ])
#
#         np.testing.assert_almost_equal(expected, targets)
#
#
# class SarsaTargetArrayCalculatorTests(unittest.TestCase):
#     def setUp(self):
#         mock_system = MockSystem()
#         self.calculator = build_sarsa_target_array_calculator(mock_system, discount_factor=0.9)
#
#     def test_get_target_array1(self):
#         states = [MockState1(), MockState2A(), MockState3()]
#         rewards = [11, 33]
#         actions = [0, 0]
#         episode = Episode(states, actions, rewards)
#
#         targets = self.calculator.get_target_matrix(episode)
#
#         expected = np.array([
#             11 + 0.9 * 1.0,
#             33 + 0.9 * 0.0
#         ])
#
#         np.testing.assert_almost_equal(expected, targets)
#
#     def test_get_target_array2(self):
#         states = [MockState1(), MockState2B(), MockState3()]
#         rewards = [22, 33]
#         actions = [1, 0]
#         episode = Episode(states, actions, rewards)
#
#         targets = self.calculator.get_target_matrix(episode)
#
#         expected = np.array([
#             22.0 + 0.9 * 0,
#             33.0 + 0.9 * 0
#         ])
#
#         np.testing.assert_array_almost_equal(expected, targets)
#
#
# class VectorizedStateMachineTargetArrayCalculatorTest(unittest.TestCase):
#
#     def setUp(self):
#
#         action_target_calculator = MagicMock()
#         rl_system = MagicMock(
#             num_internal_states=2,
#             num_actions=[2, 2]
#         )
#
#         self.calculator = VectorizedStateMachineTargetArrayCalculator(
#             rl_system,
#             action_target_calculator,
#         )
#
#     def test_get_state_targets(self):
#
#         external_state = MagicMock()
#         int_ext_state = IntExtState(0, external_state)
#
#         targets = self.calculator.get_state_targets(int_ext_state)
#
#         self.assertEqual(2, len(targets))
#
#     def test_get_target_array(self):
#
#         external_state1 = MagicMock()
#         external_state2 = MagicMock()
#
#         external_states = [external_state1, external_state2]
#
#         targets = self.calculator.get_target_matrix(external_states)
#
#         self.assertEqual(2, len(targets))
#         self.assertEqual((2, 2), targets[0].shape)
#         self.assertEqual((2, 2), targets[1].shape)


if __name__ == '__main__':
    unittest.main()
