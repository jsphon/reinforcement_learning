import tensorflow as tf


class LineWorldState(object):

    def __init__(self, position, num_positions=10):
        self.num_positions=num_positions
        self.position = position

    def as_vector(self):
        return tf.one_hot(self.position, self.num_positions, dtype=tf.float32)
