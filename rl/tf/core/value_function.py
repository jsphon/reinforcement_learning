"""
Tensorflow based Value Functions
"""

import tensorflow as tf

from rl.lib.timer import Timer

class ValueFunctionBuilder(object):
    def __init__(self,
                 input_shape,
                 hidden_shape,
                 output_shape,
                 use_one_hot_input_transform=False,
                 custom_input_transform=None
                 ):

        self.input_shape = input_shape
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape

        self.use_one_hot_input_transform = use_one_hot_input_transform

        if use_one_hot_input_transform:
            assert custom_input_transform is None
            self.custom_input_transform = None
        else:
            self.custom_input_transform = custom_input_transform

        weights = [weight_variable((input_shape, hidden_shape[0]), name='hidden_0')]
        weights += [weight_variable([hidden_shape[i], hidden_shape[i + 1]], name='hidden_weight_%i'%(i+1)) for i in range(len(hidden_shape) - 1)]
        weights += [weight_variable((hidden_shape[-1], output_shape), name='hidden_weight_%i'%len(hidden_shape))]
        self.weights = weights

        biases = [bias_variable([s], 'hidden_bias_%i'%i) for i, s in enumerate(hidden_shape)]
        biases += [bias_variable([output_shape], name='output_bias')]
        self.biases = biases

    def vectorized_2d(self, states):
        """

        Args:
            states: rank 2 tensor

        Returns:
            rank 3 tensor

            1st and 2nd dimensions (i, j) correspond to the states from the 1st 2 dimensions of states
            the 3rd (new) dimension corresponds to the action values of state at i, j

        """
        if self.use_one_hot_input_transform:
            shape = tf.shape(states)
            reshaped_states = tf.reshape(states, [-1])
            result = self.calculate(reshaped_states)
            result = tf.reshape(result, (shape[0], shape[1], -1))
            return result
        else:
            states = tf.reshape(states, (-1, self.input_shape))
        return self.calculate(states)

    def vectorized(self, states):

        if self.use_one_hot_input_transform:
            # For 1-hot, we need rank 2, we won't get
            # much performance hit with this reshaping here as
            # we are just constructing the graph, not running it
            if states.shape.ndims == 1:
                states = tf.reshape(states, (-1, 1))

        results = tf.map_fn(self.calculate, states, dtype=tf.float32)
        shape = (tf.shape(states)[0], self.output_shape)
        results = tf.reshape(results, shape)
        return results

    def calculate(self, x):

        if self.use_one_hot_input_transform:
            yi = tf.one_hot(x, depth=self.input_shape, dtype=tf.float32)
            if x.shape.ndims == 0:
                # when x is a scalar
                yi = tf.reshape(yi, (1, -1))
        elif self.custom_input_transform:
            yi = self.custom_input_transform(x)
        else:
            yi = x

        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            yi = tf.nn.relu(tf.matmul(yi, w) + b)

        y = tf.matmul(yi, self.weights[-1]) + self.biases[-1]
        return y

    def train_loop(self,
                   x,
                   y,
                   num_steps=10,
                   learning_rate=0.01
                   ):

        i = tf.constant(0)

        def cond(counter):
            return tf.less(counter, num_steps)

        def body(counter):
            train_op = self.train_op(x, y, learning_rate=learning_rate)
            with tf.control_dependencies([train_op]):
                return counter + 1

        return tf.while_loop(cond, body, [i])

    def repeat_n_times(self, op, n):
        i = tf.constant(0)

        def cond(counter):
            return tf.less(counter, n)

        def body(counter):
            with tf.control_dependencies([op]):
                return counter + 1

            return tf.while_loop(cond, body, [i])

        return tf.while_loop(cond, body, [i])

    def train_op(self, x, y, *args, **kwargs):
        with tf.variable_scope('train_op', reuse=tf.AUTO_REUSE):
            y_snap = tf.get_variable(
                'y_snap',
                shape=y.shape,
                initializer=tf.constant_initializer(0),
                trainable=False,
            )
        print('Making train op with %s' % str(y_snap))
        assign_op = tf.assign(y_snap, y)
        loss = self.squared_loss(x, y_snap)
        with tf.control_dependencies([assign_op]):
            with Timer('Making train op'):
                # train_ops = [
                #     tf.train.GradientDescentOptimizer(*args, **kwargs).minimize(loss=loss)
                #     for _ in range(2)]
                train_op = tf.train.GradientDescentOptimizer(*args, **kwargs).minimize(loss=loss)
                #train_ops = self.repeat_n_times(train_op, 1000)
        return tf.group(assign_op, train_op)

    def squared_loss(self, x, y):
        y_hat = self.calculate(x)
        loss = squared_loss(y_hat, y)
        return loss


class ValueFunctionTrainer(object):

    def __init__(self, x, y, value_function):
        self.x = x
        self.y = y
        self.y_snapshot = tf.convert_to_tensor(self.y)
        self.value_function = value_function
        self.loss = self.get_loss()

    def refresh_y_snapshot(self):
        self.y_snapshot = tf.convert_to_tensor(self.y)

    def refresh_loss(self):
        self.loss = self.get_loss()

    def get_loss(self):
        return self.value_function.squared_loss(self.x, self.y_snapshot)

    def train_op(self, *args, **kwargs):
        # with tf.variable_scope('train_op', reuse=tf.AUTO_REUSE):
        #     y_snap = tf.get_variable(
        #         'y_snap',
        #         shape=y.shape,
        #         initializer=tf.constant_initializer(0),
        #         trainable=False,
        #     )
        # print('Making train op with %s' % str(y_snap))
        # assign_op = tf.assign(y_snap, y)
        # loss = self.squared_loss(x, y_snap)
        #loss = self.get_loss()
        # with tf.control_dependencies([assign_op]):
        #     with Timer('Making train op'):
                # train_ops = [
                #     tf.train.GradientDescentOptimizer(*args, **kwargs).minimize(loss=loss)
                #     for _ in range(2)]
                #train_op = tf.train.GradientDescentOptimizer(*args, **kwargs).minimize(loss=loss)
                #train_ops = self.repeat_n_times(train_op, 1000)
        #return tf.group(assign_op, train_ops)
        train_op = tf.train.GradientDescentOptimizer(*args, **kwargs).minimize(loss=self.loss)
        return train_op


def squared_loss(y0, y1):
    squared_deltas = tf.square(y1 - y0)
    loss = tf.reduce_sum(squared_deltas)
    return loss


def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


if __name__=='__main__':

    import numpy as np

    nx = np.arange(10)
    tx = tf.Variable(nx, dtype=tf.int32, trainable=False)

    ny_true = np.array([
        [8, 8, 9, 10, 9, 8, 7, 6, 5, 4],
        [9, 10, 7, 8, 7, 6, 5, 4, 3, 2],

    ]
    ).T

    ty_true = tf.Variable(ny_true, dtype=tf.float32, trainable=False)

    builder = ValueFunctionBuilder(10, [50, 20], 2, use_one_hot_input_transform=True)
    yhat = builder.calculate(tx)

    trainer = ValueFunctionTrainer(tx, ty_true, builder)

    train_op = trainer.train_op(learning_rate=0.01)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    print(trainer.get_loss().eval())
    print(yhat.eval())

    for _ in range(1000):
        sess.run(train_op)

    print(trainer.get_loss().eval())



    print(yhat.eval())