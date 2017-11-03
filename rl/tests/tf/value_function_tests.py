import logging

logging.getLogger("tensorflow").setLevel(logging.WARNING)

import unittest
import numpy as np
import tensorflow as tf

from rl.tf.value_function import ValueFunctionBuilder


class MyTestCase(unittest.TestCase):

    def test_something(self):
        builder = ValueFunctionBuilder(
            input_shape=10,
            hidden_shape=[8, 6],
            output_shape=3,
        )

        nx = np.arange(10).reshape((1, 10))
        tx = tf.Variable(nx, dtype=tf.float32)

        y = builder.build(tx)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            result = sess.run(y)

        self.assertIsInstance(result, np.ndarray)


if __name__ == '__main__':
    unittest.main()
