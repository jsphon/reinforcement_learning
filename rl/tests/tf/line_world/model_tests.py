import unittest
import tensorflow as tf


import logging

#logging.getLogger("tensorflow").setLevel(logging.WARNING)


from rl.tf.environments.line_world.model import LineWorldModel
from rl.tf.environments.line_world.state import LineWorldState


class MyTestCase(unittest.TestCase):

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
