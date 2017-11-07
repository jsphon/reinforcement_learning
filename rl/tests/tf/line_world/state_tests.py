import unittest

import logging

logging.getLogger("tensorflow").setLevel(logging.WARNING)

import tensorflow as tf

from rl.tf.environments.line_world.state import LineWorldState


class MyTestCase(unittest.TestCase):

    def test_as_vector(self):

        test_data = (
            (0, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            (1, [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
            (2, [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
            (3, [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
            (4, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
        )

        for position, expected in test_data:
            t_position = tf.Variable(position)
            state = LineWorldState(t_position)
            vector = state.as_vector()

            with tf.Session() as sess:
                init_global_variables(sess)
                result = sess.run(vector)

            self.assertEqual(expected, result.tolist())


def init_global_variables(sess):
    sess.run(tf.global_variables_initializer())


if __name__ == '__main__':
    unittest.main()
