import unittest

import tensorflow as tf
from mock import MagicMock
from rl.tests.tf.utils import evaluate_tensor
from rl.tf.core.learning.target_array_calculator import ModelBasedTargetArrayCalculator


class ModelBasedTargetArrayCalculatorTests(unittest.TestCase):

    def test_get_state_action_target(self):

        t_state = tf.Variable('state')
        t_next_state = tf.Variable('next_state')
        t_action = tf.Variable('action')
        t_reward = tf.Variable(2.0)
        t_target = tf.Variable(3.0)
        t_next_state_action_values = tf.Variable([10.0, 20.0])

        def apply_action(state, action):
            if state == t_state and action == t_action:
                return t_next_state

        def is_terminal(state):
            if state == t_next_state:
                return tf.Variable(True)

        def reward_function(state, action, next_state):
            if state == t_state and action == t_action and next_state == t_next_state:
                return t_reward

        def action_value_function(state):
            if state==t_next_state:
                return t_next_state_action_values

        rl_system = MagicMock()
        rl_system.model.apply_action = apply_action
        rl_system.model.is_terminal = is_terminal
        rl_system.reward_function = reward_function
        rl_system.action_value_function = action_value_function

        def calculate(reward, next_state_action_values):
            if reward==t_reward and next_state_action_values==t_next_state_action_values:
                return t_target

        action_target_calculator = MagicMock()
        action_target_calculator.calculate = calculate

        calc = ModelBasedTargetArrayCalculator(rl_system, action_target_calculator)
        target = calc.get_state_action_target(t_state, t_action)

        actual = evaluate_tensor(target)
        self.assertEqual(2.0, actual)




if __name__ == '__main__':
    unittest.main()
