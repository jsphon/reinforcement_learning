import logging

logging.getLogger("tensorflow").setLevel(logging.WARNING)

import unittest
import numpy as np
import tensorflow as tf
from rl.lib.timer import Timer

from rl.tf.value_function import ValueFunctionBuilder, squared_loss

TRAIN_STEPS = 1000

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

    def test_fitting(self):
        '''
        Test that the model is flexible enough to fit
        our true value function
        Returns:
        '''

        builder = ValueFunctionBuilder(
            input_shape=10,
            hidden_shape=[8, 6],
            output_shape=3,
        )

        nx = np.eye(10)
        tx = tf.Variable(nx, dtype=tf.float32)

        ny_true = np.array([
            [9, 0, 9, 8, 7, 6, 4, 3, 2, 1],
            [9, 8, 7, 6, 5, 4, 3, 2, 3, 4],
            [1, 2, 3, 4, 5, 6, 7, 6, 5, 4],
        ]
        ).T

        ty_true = tf.Variable(ny_true, dtype=tf.float32, trainable=False)
        y = builder.build(tx)
        loss = squared_loss(y, ty_true)

        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss=loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            initial_loss, initial_y = sess.run([loss, y])

            with Timer('Training using Python loop'):
                for _ in range(TRAIN_STEPS):
                    sess.run(train_step)

                final_loss, final_y, final_ty_true = sess.run([loss, y, ty_true])

        print('initial loss : %0.2f' % initial_loss)
        print('initial y : %s' % str(initial_y))

        print('final loss : %0.2f' % final_loss)
        print('final y : %s' % str(final_y))


    def test_fitting2(self):
        '''
        Test that the model is flexible enough to fit
        our true value function
        Returns:
        '''

        builder = ValueFunctionBuilder(
            input_shape=10,
            hidden_shape=[8, 6],
            output_shape=3,
        )

        nx = np.eye(10)
        tx = tf.Variable(nx, dtype=tf.float32)

        ny_true = np.array([
            [9, 0, 9, 8, 7, 6, 4, 3, 2, 1],
            [9, 8, 7, 6, 5, 4, 3, 2, 3, 4],
            [1, 2, 3, 4, 5, 6, 7, 6, 5, 4],
        ]
        ).T

        ty_true = tf.Variable(ny_true, dtype=tf.float32, trainable=False)
        y = builder.build(tx)
        loss = squared_loss(y, ty_true)

        i = tf.constant(0)

        def cond(i):
            return tf.less(i, TRAIN_STEPS)

        def body(i):
            body_y = builder.build(tx)
            body_loss = squared_loss(body_y, ty_true)
            train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss=body_loss)

            with tf.control_dependencies([train_step]):
                return i+1
        r = tf.while_loop(cond, body, [i])

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            initial_loss, initial_y = sess.run([loss, y])

            with Timer('Training using tf.while_loop'):
                sess.run(r)

            final_loss, final_y = sess.run([loss, y])

        print('initial loss : %0.2f' % initial_loss)
        print('initial y : %s' % str(initial_y))

        print('final loss : %0.2f' % final_loss)
        print('final y : %s' % str(final_y))


if __name__ == '__main__':
    unittest.main()
