import numpy as np
import tensorflow as tf


class LineWorldState(object):

    def __init__(self, position, num_positions=10):
        self.num_positions=num_positions
        self.t_position = tf.Variable(position, dtype=tf.int32)

    def as_vector(self):
        arr = np.zeros(self.num_positions, dtype=np.float32)
        x = tf.Variable(arr, dtype=tf.float32)
        b = tf.scatter_update(x, [self.t_position], [1])
        return b
