import unittest

import numpy as np
import tensorflow as tf
from mock import MagicMock
from rl.tests.tf.utils import evaluate_tensor
from rl.tf.core.learning.target_array_calculator import ModelBasedTargetArrayCalculator

t_state = tf.Variable('state', name='non_terminal_state')
t_next_state = tf.Variable('next_state')
t_terminal_state = tf.Variable('terminal', name='terminal_state')

t_action_to_live = tf.Variable(0, name='action_to_live')
t_action_to_terminal = tf.Variable(1, name='action_to_terminal')

t_live_reward = tf.Variable(2.0, name='live_reward')
t_terminal_reward = tf.Variable(4.0, name='terminal_reward')

t_live_target = tf.Variable(3.0)
t_terminal_target = tf.Variable(-1.0, name='terminal_target')
t_next_state_action_values = tf.Variable([10.0, 20.0])


class RewardFunction(object):

    def action_reward(self, state, action, next_state):
        """
        Receives state, action, state as input, hence the name
        Args:
            state:
            action:
            next_state:

        Returns:

        """

        to_live = tf.reduce_all(
            tf.stack([
                tf.equal(t_state, state),
                tf.equal(t_action_to_live, action),
                tf.equal(t_next_state, next_state),
            ])
        )

        to_terminal = tf.reduce_all(
            tf.stack([
                tf.equal(t_state, state),
                tf.equal(t_action_to_terminal, action),
                tf.equal(t_next_state, t_terminal_state),
            ])
        )

        return tf.case({
            to_live: lambda: t_live_reward,
            to_terminal: lambda: t_terminal_reward,
            },
            exclusive=True)

    def state_rewards(self, state, next_states):

        live_reward = self.action_reward(state, t_action_to_live, next_states[0])
        terminal_reward = self.action_reward(state, t_action_to_terminal, next_states[1])
        return tf.stack([live_reward, terminal_reward])


class RewardFunctionTests(unittest.TestCase):

    def test_state_rewards(self):

        next_states = tf.stack([t_next_state, t_terminal_state])

        print('Next states are %s' % str(evaluate_tensor(next_states)))
        rf = RewardFunction()
        state_rewards = rf.state_rewards(t_state, next_states)

        actual = evaluate_tensor(state_rewards)
        desired = np.array([2.0, 4.0])

        np.testing.assert_array_equal(actual, desired)



class ModelBasedTargetArrayCalculatorTests(unittest.TestCase):

    def setUp(self):



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
                return t_live_target
            elif reward == t_terminal_reward:
                # Should not get here
                return t_terminal_target

        action_target_calculator = MagicMock()
        action_target_calculator.calculate = calculate

        self.calc = ModelBasedTargetArrayCalculator(rl_system, action_target_calculator)
        self.t_state = t_state
        self.t_action_to_live = t_action_to_live
        self.t_action_to_terminal = t_action_to_terminal

    def test_get_state_action_target_live(self):

        target = self.calc.get_state_action_target(self.t_state, self.t_action_to_live)
        actual = evaluate_tensor(target)
        self.assertEqual(3.0, actual)

    def test_get_state_action_target_terminal(self):

        target = self.calc.get_state_action_target(self.t_state, self.t_action_to_terminal)
        actual = evaluate_tensor(target)
        self.assertEqual(4.0, actual)

    def test_get_state_targets(self):

        targets = self.calc.get_state_targets(self.t_state)
        actual = evaluate_tensor(targets)

        desired = np.array([3.0, 4.0])
        np.testing.assert_array_equal(actual, desired)




if __name__ == '__main__':
    unittest.main()
