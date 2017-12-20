import unittest


import logging

logging.getLogger("tensorflow").setLevel(logging.WARNING)

import numpy as np
import tensorflow as tf
from rl.tf.core.learning.target_array_calculator import ModelBasedTargetArrayCalculator
from rl.tf.core.learning.action_target_calculator import QLearningActionTargetCalculator
from rl.tf.environments.line_world.system import LineWorldSystem
from rl.tf.environments.line_world.constants import TARGET
from rl.lib.timer import Timer


class ModelBasedTargetArrayCalculatorTests(tf.test.TestCase):

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

        t_state = tf.constant(0, dtype=tf.int32)

        lws = LineWorldSystem()
        action_target_calculator = QLearningActionTargetCalculator(lws)
        calculator = ModelBasedTargetArrayCalculator(lws, action_target_calculator)

        t_action_values = lws.action_value_function.calculate(t_state)
        t_targets = calculator.get_state_targets(t_state)

        t_target0 = calculator.get_state_action_target(t_state, 0)
        t_target1 = calculator.get_state_action_target(t_state, 1)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            a_action_values, a_targets = sess.run([t_action_values, t_targets])
            a_target0, a_target1 = sess.run([t_target0, t_target1])

        np.testing.assert_almost_equal(a_targets[0], a_target0)
        np.testing.assert_almost_equal(a_targets[1], a_target1)

    def test_get_states_targets_vectorized(self):

        a_states = np.arange(10)
        t_states = tf.constant(a_states, dtype=tf.int32)

        lws = LineWorldSystem()
        action_target_calculator = QLearningActionTargetCalculator(lws)

        calculator = ModelBasedTargetArrayCalculator(lws, action_target_calculator)

        t_targets = calculator.get_states_targets_vectorized(t_states)
        t_expected_targets = calculator.get_states_targets(t_states)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            a_targets = sess.run(t_targets)
            a_expected_targets = sess.run(t_expected_targets)

        self.assertEqual((10, 2), a_targets.shape)
        self.assertAllClose(a_expected_targets, a_targets)

    def test_get_states_targets_vectorized_perf(self):

        a_states = np.arange(10)
        t_states = tf.constant(a_states, dtype=tf.int32)

        lws = LineWorldSystem()
        action_target_calculator = QLearningActionTargetCalculator(lws)

        calculator = ModelBasedTargetArrayCalculator(lws, action_target_calculator)

        t_targets = calculator.get_states_targets_vectorized(t_states)
        t_expected_targets = calculator.get_states_targets(t_states)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            with Timer('vectorized'):
                for _ in range(100):
                    a_targets = sess.run(t_targets)

            with Timer('map'):
                for _ in range(100):
                    a_expected_targets = sess.run(t_expected_targets)

        self.assertEqual((10, 2), a_targets.shape)
        self.assertAllClose(a_expected_targets, a_targets)

    def test_get_states_targets(self):
        a_states = np.arange(10)
        t_states = tf.constant(a_states, dtype=tf.int32)

        lws = LineWorldSystem()
        action_target_calculator = QLearningActionTargetCalculator(lws)

        calculator = ModelBasedTargetArrayCalculator(lws, action_target_calculator)

        t_targets = calculator.get_states_targets(t_states)

        expected_targets = [calculator.get_state_targets(tf.Variable(x)) for x in range(10)]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            a_targets = sess.run(t_targets)
            a_expected_targets = sess.run(expected_targets)

        print('targets are %s' % str(a_targets))
        print('expected targets are %s' % str(a_expected_targets))

        for expected, actual in zip(a_targets, a_expected_targets):
            np.testing.assert_almost_equal(expected, actual)

    def xtest_get_states_targets_with_train_op(self):
        a_states = np.arange(10)
        t_states = tf.constant(a_states, dtype=tf.int32)

        lws = LineWorldSystem()
        action_target_calculator = QLearningActionTargetCalculator(lws)

        calculator = ModelBasedTargetArrayCalculator(lws, action_target_calculator)

        t_targets = calculator.get_states_targets(t_states)
        t_action_values = lws.action_value_function.vectorized(t_states)

        expected_targets = [calculator.get_state_targets(tf.Variable(x)) for x in range(10)]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            a_targets = sess.run(t_targets)
            a_expected_targets = sess.run(expected_targets)
            a_action_values = sess.run(t_action_values)

            print('targets are %s' % str(a_targets))
            print('expected targets are %s' % str(a_expected_targets))
            print('action values are %s' % str(a_action_values))

            expected_target = -1.0 + a_action_values[0].max()
            np.testing.assert_almost_equal(expected_target, a_targets[0, 0])

            for expected, actual in zip(a_targets, a_expected_targets):
                np.testing.assert_almost_equal(expected, actual)

            #for _ in range(10):
            train_op = lws.action_value_function.train_op(t_states, t_targets, learning_rate=0.001)
            sess.run(train_op)

            a_targets = sess.run(t_targets)
            a_expected_targets = sess.run(expected_targets)
            a_action_values = sess.run(t_action_values)

            print('targets are %s' % str(a_targets))
            print('expected targets are %s' % str(a_expected_targets))
            print('action values are %s' % str(a_action_values))

            expected_target = -1.0 + a_action_values[0].max()
            np.testing.assert_almost_equal(expected_target, a_targets[0, 0])

            for expected, actual in zip(a_targets, a_expected_targets):
                np.testing.assert_almost_equal(expected, actual)


if __name__ == '__main__':
    unittest.main()
