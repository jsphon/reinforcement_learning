import unittest
import tensorflow as tf
from rl.tf.environments.line_world.constants import MOVE_LEFT, MOVE_RIGHT

import logging

#logging.getLogger("tensorflow").setLevel(logging.WARNING)


from rl.tf.environments.line_world.model import LineWorldModel
from rl.tf.environments.line_world.state import LineWorldState


class MyTestCase(unittest.TestCase):

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
            state = LineWorldState(position=t_position)
            model = LineWorldModel()

            new_state = model.move_left(state)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                result = sess.run(new_state.position)

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
            state = LineWorldState(position=t_position)
            model = LineWorldModel()

            new_state = model.move_right(state)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                result = sess.run(new_state.position)

            self.assertEqual(p1, result)




if __name__ == '__main__':
    unittest.main()
