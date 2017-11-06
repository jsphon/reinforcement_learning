import logging

logging.getLogger("tensorflow").setLevel(logging.WARNING)

import unittest
import numpy as np
import tensorflow as tf
from rl.lib.timer import Timer

from rl.tf.value_function import ValueFunctionBuilder, squared_loss

TRAIN_STEPS = 10000
LEARNING_RATE = 0.001


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.builder = ValueFunctionBuilder(
            input_shape=10,
            hidden_shape=[8, 6],
            output_shape=3,
        )

        nx = np.eye(10)
        self.tx = tf.Variable(nx, dtype=tf.float32, trainable=False)

        ny_true = np.array([
            [9, 0, 9, 8, 7, 6, 4, 3, 2, 1],
            [9, 8, 7, 6, 5, 4, 3, 2, 3, 4],
            [1, 2, 3, 4, 5, 6, 7, 6, 5, 4],
        ]
        ).T

        self.ty_true = tf.Variable(ny_true, dtype=tf.float32, trainable=False)

    def test_something(self):

        y = self.builder.build(self.tx)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            result = sess.run(y)

        self.assertIsInstance(result, np.ndarray)

    def test_train_op(self):
        '''
        Test that the model is flexible enough to fit
        our true value function
        Returns:
        '''

        y = self.builder.build(self.tx)
        loss = squared_loss(y, self.ty_true)

        train_op = self.builder.train_op(self.tx, self.ty_true, learning_rate=LEARNING_RATE)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            initial_loss, initial_y = sess.run([loss, y])

            with Timer('Training using Python loop with %i steps' % TRAIN_STEPS):
                for _ in range(TRAIN_STEPS):
                    sess.run(train_op)

                final_loss, final_y, final_ty_true = sess.run([loss, y, self.ty_true])

        print('initial loss : %0.2f' % initial_loss)
        print('initial y : %s' % str(initial_y))

        print('final loss : %0.2f' % final_loss)
        print('final y : %s' % str(final_y))

        self.assertLess(final_loss, initial_loss)

    def test_train_loop(self):
        '''
        Test that the model is flexible enough to fit
        our true value function
        Returns:
        '''

        y = self.builder.build(self.tx)
        loss = squared_loss(y, self.ty_true)

        train_loop = self.builder.train_loop(self.tx,
                                             self.ty_true,
                                             learning_rate=LEARNING_RATE,
                                             num_steps=TRAIN_STEPS
                                             )

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            initial_loss, initial_y = sess.run([loss, y])

            with Timer('Training using train_loop %i steps' % TRAIN_STEPS):
                sess.run(train_loop)

                final_loss, final_y, final_ty_true = sess.run([loss, y, self.ty_true])

        print('initial loss : %0.2f' % initial_loss)
        print('initial y : %s' % str(initial_y))

        print('final loss : %0.2f' % final_loss)
        print('final y : %s' % str(final_y))

        self.assertLess(final_loss, initial_loss)


if __name__ == '__main__':
    unittest.main()
