import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)

import unittest
import numpy as np
import tensorflow as tf

from rl.tf.environments.line_world.constants import MOVE_LEFT, MOVE_RIGHT
from rl.tf.environments.line_world.model import LineWorldModel
from rl.tf.environments.line_world.constants import TARGET
from rl.tests.tf.utils import evaluate_tensor


class MyTestCase(unittest.TestCase):

    def test_are_states_terminal(self):
        model = LineWorldModel()

        positions = tf.Variable([TARGET-1, TARGET, TARGET+1])
        are_terminal = model.are_states_terminal(positions)

        result = evaluate_tensor(are_terminal)
        expected = np.array([False, True, False])
        np.testing.assert_array_equal(expected, result)

    def test_is_terminal_True(self):
        model = LineWorldModel()

        is_terminal = model.is_terminal(TARGET)

        with tf.Session() as sess:
            tf.global_variables_initializer()
            actual = sess.run(is_terminal)

        self.assertTrue(actual)

    def test_is_terminal_False(self):
        model = LineWorldModel()

        is_terminal = model.is_terminal(TARGET+1)

        with tf.Session() as sess:
            tf.global_variables_initializer()
            actual = sess.run(is_terminal)

        self.assertFalse(actual)

    def test_apply_actions(self):

        model = LineWorldModel()

        t_state = tf.Variable(1)
        t_next_states = model.apply_actions(t_state)
        result = evaluate_tensor(t_next_states)

        expected = np.array([0, 2])
        np.testing.assert_array_equal(expected, result)

    def test_apply_action(self):

        test_data = (
            (MOVE_LEFT, 0, 0),
            (MOVE_LEFT, 1, 0),
            (MOVE_LEFT, 2, 1),
            (MOVE_LEFT, 3, 2),
            (MOVE_LEFT, 5, 4),
            (MOVE_LEFT, 9, 8),
            (MOVE_RIGHT, 1, 2),
            (MOVE_RIGHT, 2, 3),
            (MOVE_RIGHT, 3, 4),
            (MOVE_RIGHT, 5, 6),
            (MOVE_RIGHT, 9, 9)
        )

        for action, p0, p1 in test_data:
            t_position = tf.Variable(p0)
            model = LineWorldModel()

            new_position = model.apply_action(t_position, action)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                result = sess.run(new_position)

            self.assertEqual(p1, result)

    def test_move_left(self):

        test_data = (
            (0, 0),
            (1, 0),
            (2, 1),
            (3, 2),
            (5, 4),
            (9, 8)
        )

        for p0, p1 in test_data:
            t_position = tf.Variable(p0)
            model = LineWorldModel()

            new_position = model.move_left(t_position)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                result = sess.run(new_position)

            self.assertEqual(p1, result)

    def test_move_right(self):

        test_data = (
            (1, 2),
            (2, 3),
            (3, 4),
            (5, 6),
            (9, 9)
        )

        for p0, p1 in test_data:
            t_position = tf.Variable(p0)
            model = LineWorldModel()

            new_position = model.move_right(t_position)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                result = sess.run(new_position)

            self.assertEqual(p1, result)




if __name__ == '__main__':
    unittest.main()
