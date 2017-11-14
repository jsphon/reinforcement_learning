import unittest

import tensorflow as tf
from mock import MagicMock
from rl.tests.tf.utils import evaluate_tensor
from rl.tf.core.learning.target_array_calculator import ModelBasedTargetArrayCalculator


class ModelBasedTargetArrayCalculatorTests(unittest.TestCase):

    def test_get_state_action_target(self):

        t_state = tf.Variable('state', name='non_terminal_state')
        t_next_state = tf.Variable('next_state')
        t_terminal_state = tf.Variable('terminal', name='terminal_state')

        t_action_to_live = tf.Variable('action', name='action_to_live')
        t_action_to_terminal = tf.Variable('terminal', name='action_to_terminal')

        t_live_reward = tf.Variable(2.0, name='live_reward')
        t_terminal_reward = tf.Variable(4.0, name='terminal_reward')
        t_target = tf.Variable(3.0)
        t_terminal_target = tf.Variable(-1.0, name='terminal_target')
        t_next_state_action_values = tf.Variable([10.0, 20.0])

        def apply_action(state, action):
            if state == t_state and action == t_action_to_live:
                return t_next_state
            elif state == t_state and action == t_action_to_terminal:
                return t_terminal_state

        def is_terminal(state):
            predicate = tf.equal(state, t_terminal_state)
            return predicate

        def reward_function(state, action, next_state):
            if state == t_state and action == t_action_to_live and next_state == t_next_state:
                return t_live_reward
            elif state == t_state and action == t_action_to_terminal and next_state == t_terminal_state:
                return t_terminal_reward

        def action_value_function(state):
            if state == t_next_state:
                return t_next_state_action_values

        rl_system = MagicMock()
        rl_system.model.apply_action = apply_action
        rl_system.model.is_terminal = is_terminal
        rl_system.reward_function = reward_function
        rl_system.action_value_function = action_value_function

        def calculate(reward, next_state_action_values):
            if reward == t_live_reward and next_state_action_values == t_next_state_action_values:
                return t_target
            elif reward == t_terminal_reward:
                # Should not get here
                return t_terminal_target

        action_target_calculator = MagicMock()
        action_target_calculator.calculate = calculate

        calc = ModelBasedTargetArrayCalculator(rl_system, action_target_calculator)

        target = calc.get_state_action_target(t_state, t_action_to_live)
        actual = evaluate_tensor(target)
        self.assertEqual(3.0, actual)

        target = calc.get_state_action_target(t_state, t_action_to_terminal)
        actual = evaluate_tensor(target)
        self.assertEqual(4.0, actual)


if __name__ == '__main__':
    unittest.main()
