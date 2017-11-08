import tensorflow as tf

from rl.tf.environments.line_world.constants import MOVE_LEFT, MOVE_RIGHT
from rl.tf.environments.line_world.state import LineWorldState


class LineWorldModel(object):

    def __init__(self, num_positions=10):
        self.num_positions = num_positions

    def move_left(self, position):
        return tf.maximum(position - 1, tf.Variable(0, dtype=tf.int32))

    def move_right(self, position):
        return tf.minimum(position + 1, tf.Variable(self.num_positions - 1, dtype=tf.int32))

    def apply_action(self, position, action):
        cond = tf.equal(action, MOVE_LEFT)
        new_state = tf.cond(cond, lambda: self.move_left(position), lambda: self.move_right(position))
        return new_state
