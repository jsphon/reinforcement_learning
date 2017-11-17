import unittest


import logging

logging.getLogger("tensorflow").setLevel(logging.WARNING)

import numpy as np
import tensorflow as tf
from mock import MagicMock
from rl.tests.tf.utils import evaluate_tensor
from rl.tf.core.learning.target_array_calculator import ModelBasedTargetArrayCalculator


STATE = 0
NEXT_STATE = 1
TERMINAL = 2

t_state = tf.Variable(STATE, name='non_terminal_state')
t_next_state = tf.Variable(NEXT_STATE, name='next_state')
t_terminal_state = tf.Variable(TERMINAL, name='terminal_state')

t_action_to_live = tf.Variable(0, name='action_to_live')
t_action_to_terminal = tf.Variable(1, name='action_to_terminal')

t_live_reward = tf.Variable(2.0, name='live_reward')
t_terminal_reward = tf.Variable(4.0, name='terminal_reward')

t_live_target = tf.Variable(3.0, name='live_target')
t_terminal_target = tf.Variable(-1.0, name='terminal_target')
t_next_state_action_values = tf.Variable([10.0, 20.0], name='next_state_action_values')


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

        print(t_next_state.graph)
        print(next_state.graph)
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


class ActionTargetCalculator(object):

    def vectorized(self, rewards, next_states_action_values):
        t0 = self.calculate(rewards[0], next_states_action_values[0])
        t1 = self.calculate(rewards[1], next_states_action_values[1])
        return tf.stack([t0, t1])

    def calculate(self, reward, next_state_action_values):

        live_target = tf.reduce_all(
            tf.stack([
                tf.equal(reward, t_live_reward),
                tf.reduce_all(tf.equal(next_state_action_values, t_next_state_action_values))
            ])
        )

        result = tf.case(
            {
                live_target: lambda: t_live_target
            },
            default=lambda: t_terminal_target,
            exclusive=True,
        )

        return result


class Model(object):

    def apply_actions(self, state):

        s0 = self.apply_action(state, 0)
        s1 = self.apply_action(state, 1)

        return tf.stack([s0, s1])

    def apply_action(self, state, action):
        when_next_state = tf.reduce_all(
            tf.stack([
                tf.equal(state, t_state),
                tf.equal(action, t_action_to_live)
            ])
        )

        when_terminal_state = tf.reduce_all(
            tf.stack([
                tf.equal(state, t_state),
                tf.equal(action, t_action_to_terminal),
            ])
        )

        result = tf.case({
            when_next_state: lambda: t_next_state,
            when_terminal_state: lambda: t_terminal_state
        },
            default=lambda: t_state,
            exclusive=True
        )

        return result

    def are_states_terminal(self, states):
        t0 = self.is_terminal(states[0])
        t1 = self.is_terminal(states[1])
        return tf.stack([t0, t1])
        #return tf.map_fn(self.is_terminal, states)

    def is_terminal(self, state):
        predicate = tf.equal(state, t_terminal_state)
        return predicate


class ActionValueFunction(object):

    def vectorized(self, states):
        #return tf.map_fn(self.calculate, states, dtype=t_next_state_action_values.dtype)
        v0 = self.calculate(states[0])
        v1 = self.calculate(states[1])
        return tf.stack([v0, v1])


    def calculate(self, state):
        return t_next_state_action_values



class ModelBasedTargetArrayCalculatorTests(unittest.TestCase):

    def setUp(self):

        # def apply_action(state, action):
        #     when_next_state = tf.reduce_all(
        #         tf.stack([
        #             tf.equal(state, t_state),
        #             tf.equal(action, t_action_to_live)
        #         ])
        #     )
        #
        #     when_terminal_state = tf.reduce_all(
        #         tf.stack([
        #             tf.equal(state, t_state),
        #             tf.equal(action, t_action_to_terminal),
        #         ])
        #     )
        #
        #     result = tf.case({
        #         when_next_state: lambda: t_next_state,
        #         when_terminal_state: lambda: t_terminal_state
        #     },
        #         default=lambda: t_state,
        #         exclusive=True
        #     )
        #
        #     return result

        # def is_terminal(state):
        #     predicate = tf.equal(state, t_terminal_state)
        #     return predicate

        # def action_value_function(state):
        #     return t_next_state_action_values

        rl_system = MagicMock()
        rl_system.model = Model()
        #rl_system.model.apply_action = apply_action
        #rl_system.model.is_terminal = is_terminal
        rl_system.reward_function = RewardFunction()
        rl_system.action_value_function = ActionValueFunction()

        # def calculate(reward, next_state_action_values):
        #     if reward == t_live_reward and next_state_action_values == t_next_state_action_values:
        #         return t_live_target
        #     elif reward == t_terminal_reward:
        #         # Should not get here
        #         return t_terminal_target
        #     else:
        #         raise ValueError('How did we get here?')

        action_target_calculator = ActionTargetCalculator()

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
