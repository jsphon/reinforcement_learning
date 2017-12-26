import tensorflow as tf


class Trainer(object):

    def __init__(self, x, target_array_calculator, value_function, optimizer=None):
        self.x = x
        self.y = target_array_calculator.get_states_targets_vectorized(self.x)
        self.target_array_calculator = target_array_calculator
        self.value_function = value_function
        self.optimizer = optimizer or tf.train.GradientDescentOptimizer(learning_rate=0.01)

        self.y_snapshot = None
        self.loss = None
        self.train_op = None
        self.train_loop = None

        self.update_y_snapshot()
        self.update_loss()
        self.update_train_op()
        self.update_train_loop(10)

    def update_y_snapshot(self):
        self.y_snapshot = tf.convert_to_tensor(self.y)

    def update_loss(self):
        self.loss = self.squared_loss(self.x, self.y_snapshot)

    def update_train_op(self):
        self.train_op = self.optimizer.minimize(loss=self.loss)

    def update_train_loop(self, num_steps):
        i = tf.constant(0)

        def cond(counter):
            return tf.less(counter, num_steps)

        def body(counter):
            # Hmmm... Does the train op need to be defined inside here for it to work?
            with tf.control_dependencies([self.train_op]):
                return counter + 1

        self.train_loop = tf.while_loop(cond, body, [i])

    def train_once(self, sess):
        self.update_y_snapshot()
        self.update_loss()
        self.update_train_op()
        sess.run(self.train_op)


    def squared_loss(self, x, y):
        y_hat = self.value_function.calculate(x)
        loss = squared_loss(y_hat, y)
        return loss


def squared_loss(y0, y1):
    squared_deltas = tf.square(y1 - y0)
    loss = tf.reduce_sum(squared_deltas)
    return loss


def repeat_n_times(op, n):
    i = tf.constant(0)

    def cond(counter):
        return tf.less(counter, n)

    def body(counter):
        with tf.control_dependencies([op]):
            return counter + 1

        return tf.while_loop(cond, body, [i])

    return tf.while_loop(cond, body, [i])
