import numpy as np
from scipy import stats
import tensorflow as tf

from rl.tf.lib import repeat


class RepeatTestCase(tf.test.TestCase):

    def test_repeat(self):
        c = tf.Variable(0)

        def make_op():
            op = tf.assign(c, c+1)
            op = tf.Print(op, [c], message='hello')
            return tf.group(op)

        repeat_op = repeat(make_op, 10)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            r0 = sess.run(repeat_op)
            print(r0)
            result = c.eval()

        self.assertEqual(10, result)


class RepeatSlowTestCase(tf.test.TestCase):
    """
    More complete example
    """

    def test_linear_regression(self):
        n = 1000

        a = 2.0
        b = 1.0

        np.random.seed(0)
        aX = np.random.uniform(0, 10, n)
        ay = a * aX + b + np.random.standard_normal(n)

        # Check results
        slope, intercept, r_value, p_value, std_err = stats.linregress(aX, ay)

        print('slope : %0.2f' % slope)
        print('intercept : %0.2f' % intercept)

        tX = tf.Variable(aX, trainable=False)
        ty = tf.Variable(ay, trainable=False)

        va = tf.Variable(0.0, dtype=tf.float64)
        vb = tf.Variable(0.0, dtype=tf.float64)

        def get_loss():
            yhat = va * tX + vb
            loss = tf.losses.mean_squared_error(ty, yhat)
            return loss

        def get_train_op():
            # We have to create the model and loss inside the train op
            loss = get_loss()
            return tf.train.GradientDescentOptimizer(0.01).minimize(loss)

        def body():
            return get_train_op()

        n = 5000

        train_loop = repeat(body, n)

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        with self.test_session() as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(train_loop)

            opt_a = va.eval()
            opt_b = vb.eval()

            final_loss = get_loss().eval()

        print('Optimised a : %0.2f' % opt_a)
        print('Optimised b : %0.2f' % opt_b)
        print('Final Loss : %0.2f' % final_loss)

        self.assertAlmostEqual(slope, opt_a, places=3)
        self.assertAlmostEqual(intercept, opt_b, places=3)


    def test_linear_regression_sgd(self):
        """
        Test that we can fit a linear regression model using stochastic gradient descent
        Returns:

        """
        batch_size = 10
        sample_size = 10000

        a = 2.0
        b = 1.0

        np.random.seed(0)
        aX = np.random.uniform(0, 10, sample_size)
        ay = a * aX + b + np.random.standard_normal(sample_size)

        # Check results
        slope, intercept, r_value, p_value, std_err = stats.linregress(aX, ay)

        print('slope : %0.2f' % slope)
        print('intercept : %0.2f' % intercept)

        tX = tf.Variable(aX, trainable=False)
        ty = tf.Variable(ay, trainable=False)

        va = tf.Variable(0.0, dtype=tf.float64)
        vb = tf.Variable(0.0, dtype=tf.float64)

        def get_loss():
            yhat = va * tX + vb
            loss = tf.losses.mean_squared_error(ty, yhat)
            return loss

        def get_batch_loss():
            """
            Returns the loss value, given a random batch.

            Note that the batch index, model (yhat) and loss are all definied within this function. Otherwise
            the training won't work.
            Returns:

            """
            batch_idx = tf.random_uniform((batch_size,), maxval=sample_size, dtype=tf.int64)
            batch_X = tf.gather(tX, batch_idx)
            batch_y = tf.gather(ty, batch_idx)
            yhat = va * batch_X + vb
            loss = tf.losses.mean_squared_error(batch_y, yhat)
            return loss


        def get_train_op():
            # We have to create the model and loss inside the train op
            loss = get_loss()
            return tf.train.GradientDescentOptimizer(0.01).minimize(loss)

        def body():
            return get_train_op()

        n = 5000

        train_loop = repeat(body, n)

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        with self.test_session() as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(train_loop)

            opt_a = va.eval()
            opt_b = vb.eval()

            final_loss = get_loss().eval()

        print('=== SGD Results ===')
        print('Optimised/scipy a : %0.2f / %0.2f' % (opt_a, slope))
        print('Optimised/scipy b : %0.2f / %0.2f' % (opt_b, intercept))
        print('Final Loss : %0.2f' % final_loss)

        self.assertAlmostEqual(slope, opt_a, places=3)
        self.assertAlmostEqual(intercept, opt_b, places=3)