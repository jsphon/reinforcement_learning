import logging

logging.getLogger("tensorflow").setLevel(logging.WARNING)

import unittest
import numpy as np
import tensorflow as tf
from rl.lib.timer import Timer
from rl.tests.tf.utils import evaluate_tensor

from rl.tf.core.value_function import ValueFunctionBuilder, squared_loss

TRAIN_STEPS = 1000
LEARNING_RATE = 0.001


class CustomInputTransformTests(unittest.TestCase):

    def setUp(self):

        def input_transform(x):
            depth = tf.shape(x)[0]
            x = tf.one_hot(x, depth)
            return x

        self.builder = ValueFunctionBuilder(
            input_shape=10,
            hidden_shape=[8, 6],
            output_shape=3,
            custom_input_transform=input_transform,
        )

        nx = np.arange(10)
        self.tx = tf.Variable(nx, dtype=tf.int32, trainable=False)

        ny_true = np.array([
            [9, 0, 9, 8, 7, 6, 4, 3, 2, 1],
            [9, 8, 7, 6, 5, 4, 3, 2, 3, 4],
            [1, 2, 3, 4, 5, 6, 7, 6, 5, 4],
        ]
        ).T

        self.ty_true = tf.Variable(ny_true, dtype=tf.float32, trainable=False)

    def test_build(self):
        y = self.builder.calculate(self.tx)

        result = evaluate_tensor(y)
        print(result)

        self.assertIsInstance(result, np.ndarray)


class OneHotInputTransformTests(unittest.TestCase):

    def setUp(self):
        self.builder = ValueFunctionBuilder(
            input_shape=10,
            hidden_shape=[8, 6],
            output_shape=3,
            use_one_hot_input_transform=True,
        )

        nx = np.arange(10)
        self.tx = tf.Variable(nx, dtype=tf.int32, trainable=False)

        ny_true = np.array([
            [9, 0, 9, 8, 7, 6, 4, 3, 2, 1],
            [9, 8, 7, 6, 5, 4, 3, 2, 3, 4],
            [1, 2, 3, 4, 5, 6, 7, 6, 5, 4],
        ]
        ).T

        self.ty_true = tf.Variable(ny_true, dtype=tf.float32, trainable=False)

    def test_train_op(self):
        '''
        Test that the model is flexible enough to fit
        our true value function
        Returns:
        '''

        y = self.builder.calculate(self.tx)
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
        self.assertLess(final_loss, 1)

    def test_train_loop(self):
        '''
        Test that the model is flexible enough to fit
        our true value function
        Returns:
        '''

        y = self.builder.calculate(self.tx)
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
        self.assertLess(final_loss, 1)

    def test_calculate(self):

        tx = tf.Variable([1], dtype=tf.int32, trainable=False)
        y = self.builder.calculate(tx)

        result = evaluate_tensor(y)
        print(result)

        self.assertIsInstance(result, np.ndarray)

    def test_vectorized_rank1(self):

        tx = tf.Variable([0, 1, 2, 3])

        y = self.builder.vectorized(tx)

        result = evaluate_tensor(y)
        shape = tuple(evaluate_tensor(tf.shape(result)))
        self.assertEqual((4, 3), shape)

    def test_vectorized_rank2(self):

        tx = tf.Variable([0, 1, 2, 3])
        tx = tf.reshape(tx, (-1, 1))

        y = self.builder.vectorized(tx)

        result = evaluate_tensor(y)
        shape = tuple(evaluate_tensor(tf.shape(result)))
        self.assertEqual((4, 3), shape)

    def test_vectorized_values(self):
        """
        Show that vectorized returns the same values as those we would get
         if we looped through a list

        """

        positions = [0, 1, 2, 3]
        tx = tf.Variable(positions)
        tx = tf.reshape(tx, (-1, 1))

        y = self.builder.vectorized(tx)

        # There are 4 input states, and 3 actions, so we expect
        # an output of shape (4, 3)

        for position in positions:

            t_position = tf.Variable(position)
            t_expected = self.builder.calculate(t_position)

            [ay, ey] = evaluate_tensor([y, t_expected])

            np.testing.assert_array_equal(ey.ravel(), ay[position])


class OneHotInputTransformTests2(unittest.TestCase):

    def setUp(self):
        self.builder = ValueFunctionBuilder(
            input_shape=10,
            hidden_shape=[100, 100, 60, 60, 40, 20],
            output_shape=2,
            use_one_hot_input_transform=True,
        )

        nx = np.arange(10)
        self.tx = tf.Variable(nx, dtype=tf.int32, trainable=False)

        ny_true = np.array([
            [8, 8, 9, 10, 9, 8, 7, 6, 5, 4],
            [9, 10, 7, 8, 7, 6, 5, 4, 3, 2],

        ]
        ).T

        self.ty_true = tf.Variable(ny_true, dtype=tf.float32, trainable=False)

    def test_train_op(self):
        '''
        Test that the model is flexible enough to fit
        our true value function
        Returns:
        '''

        y = self.builder.calculate(self.tx)
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
        self.assertLess(final_loss, 1)

    def test_train_loop(self):
        '''
        Test that the model is flexible enough to fit
        our true value function
        Returns:
        '''

        y = self.builder.calculate(self.tx)
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

        print('initial loss : %0.2f' % initial_loss.round(2))
        print('initial y : %s' % str(initial_y.round(2)))

        print('final loss : %0.2f' % final_loss.round(2))
        print('final y : %s' % str(final_y.round(2)))

        self.assertLess(final_loss, initial_loss)
        self.assertLess(final_loss, 1)



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

    def test_calculate(self):
        y = self.builder.calculate(self.tx)

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

        y = self.builder.calculate(self.tx)
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
        self.assertLess(final_loss, 1)

    def test_train_loop(self):
        '''
        Test that the model is flexible enough to fit
        our true value function
        Returns:
        '''

        y = self.builder.calculate(self.tx)
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
        self.assertLess(final_loss, 1)


if __name__ == '__main__':
    unittest.main()
