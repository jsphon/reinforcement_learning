import logging
import unittest
import numpy as np

from rl.core.learning.target_array_calculator import build_target_array_calculator

from rl.tests.core.learning.example_state_machine_rl_system import MockStateMachineSystem, build_state
from rl.core.state import IntExtState, State, StateList

from rl.tests.core.learning.example_state_machine_rl_system import AgentLocation
from rl.tests.core.learning.example_state_machine_rl_system import AT_HOME, AT_RESTAURANT, AT_BAR, HUNGRY, SATISFIED


N = 1000

logging.basicConfig(level=logging.DEBUG)


class StateMachineTargetArrayCalculatorTests(unittest.TestCase):

    def setUp(self):
        rl_system = MockStateMachineSystem()
        self.calculator = build_target_array_calculator(
            rl_system=rl_system,
            calculator_type='modelbasedstatemachine',
            learning_algo='qlearning'
        )

    def test_get_target_matrix(self):

        s1 = AgentLocation(AT_BAR)
        s2 = AgentLocation(AT_RESTAURANT)
        lst = StateList([s1, s2])
        targets = self.calculator.get_target_matrix(lst)
        print(targets)
        self.assertEqual(2, len(targets))

        self.assertEqual((2, 3), targets[0].shape)
        self.assertEqual((2, 3), targets[1].shape)

    def test_get_state_targets(self):

        state = build_state(HUNGRY, AT_RESTAURANT)
        targets = self.calculator.get_state_targets(state)

        # rewards are 10, -1, -1
        # Action values are:
        # For action EAT=0, expected subsequent state is
        # (SATISFIED, AT_RESTAURANT) which has best value 9
        # so value is 19
        # Similarly, next targets are -1 + 3 = 2
        #                             -1 + 6 = 5

        expected = np.array([19, 2, 5])
        np.testing.assert_array_equal(expected, targets)




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


if __name__ == '__main__':
    unittest.main()
