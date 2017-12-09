import unittest


import logging

logging.getLogger("tensorflow").setLevel(logging.WARNING)

import numpy as np
import tensorflow as tf
from mock import MagicMock
from rl.tests.tf.utils import evaluate_tensor
from rl.tf.core.learning.target_array_calculator import ModelBasedTargetArrayCalculator

import numpy as np
import tensorflow as tf
from rl.tests.tf.utils import evaluate_tensor
from rl.tf.core.policy import EpsilonGreedyPolicy
from rl.tf.core.system import System
from rl.tf.core.value_function import ValueFunctionBuilder
from rl.tf.environments.line_world.model import LineWorldModel
from rl.tf.core.learning.target_array_calculator import ModelBasedTargetArrayCalculator
from rl.tf.core.learning.action_target_calculator import QLearningActionTargetCalculator
from rl.tf.environments.line_world.reward_function import RewardFunction
from rl.tf.environments.line_world.system import LineWorldSystem
from rl.tf.environments.line_world.constants import TARGET


class ModelBasedTargetArrayCalculatorTests(unittest.TestCase):

    def test_get_state_action_target(self):

        lws = LineWorldSystem()
        action_target_calculator = QLearningActionTargetCalculator(lws)

        calculator = ModelBasedTargetArrayCalculator(lws, action_target_calculator)

        t_state = tf.Variable(-1, dtype=tf.int32, name='state')
        t_next_state = lws.model.apply_action(t_state, action=0)
        t_action_values = lws.action_value_function.calculate(t_next_state)

        with tf.name_scope('test_name_scope'):
            t_target = calculator.get_state_action_target(t_state, action=0)

        tf.summary.scalar('target', t_target)
        tf.summary.scalar('state', t_state)

        merged = tf.summary.merge_all()

        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter('/tmp/tensorboard', sess.graph)
            sess.run(tf.global_variables_initializer())

            for position in range(10):

                assign_op = t_state.assign(position)
                sess.run(assign_op)

                next_state, target, action_values = sess.run([t_next_state, t_target, t_action_values])

                if next_state == TARGET:
                    expected_target = 10.0
                else:
                    expected_target = -1 + action_values.max()

                np.testing.assert_almost_equal(expected_target, target)

                target_diff = target-expected_target
                print('=== %s ===' % position)
                print('next state is %s'%str(next_state))
                print('target is %s'%str(target))
                print('expected target is %s' % str(expected_target))
                print('diff is %s'%target_diff)
                print('action_values are\n%s' % str(action_values))

                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary = sess.run(
                    merged,
                    options=run_options,
                    run_metadata=run_metadata
                )
                train_writer.add_run_metadata(run_metadata, 'position_%s'%position)
                train_writer.add_summary(summary, position)

            train_writer.close()

    def test_get_state_targets(self):

        t_state = tf.constant([0], dtype=tf.int32)

        lws = LineWorldSystem()
        action_target_calculator = QLearningActionTargetCalculator(lws)

        calculator = ModelBasedTargetArrayCalculator(lws, action_target_calculator)

        t_action_values = lws.action_value_function.calculate(t_state)
        t_targets = calculator.get_states_targets(t_state)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            a_action_values, a_targets = sess.run([t_action_values, t_targets])

        print('action values:\n%s'%str(a_action_values))

        print('targets:\n%s' % str(a_targets))

        t0 = action_target_calculator.calculate(-1.0, a_action_values[0])
        print('t0: %s' % evaluate_tensor(t0))

        np.testing.assert_equal(a_targets[0][0], a_action_values[0].max()-1.0)



    def test_xx(self):
        a_states = np.arange(10)
        t_states = tf.constant(a_states, dtype=tf.int32)

        lws = LineWorldSystem()
        action_target_calculator = QLearningActionTargetCalculator(lws)

        calculator = ModelBasedTargetArrayCalculator(lws, action_target_calculator)

        t_action_values = lws.action_value_function.calculate(t_states)
        t_targets = calculator.get_states_targets(t_states)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            a_action_values, a_targets = sess.run([t_action_values, t_targets])

        print('action values:\n%s'%str(a_action_values))

        print('targets:\n%s' % str(a_targets))

        t0 = action_target_calculator.calculate(-1.0, a_action_values[0])
        print('t0: %s' % evaluate_tensor(t0))

        np.testing.assert_equal(a_targets[0][0], a_action_values[0].max()-1.0)

if __name__ == '__main__':
    unittest.main()
