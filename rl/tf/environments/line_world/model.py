import tensorflow as tf

from rl.tf.environments.line_world.constants import MOVE_LEFT, MOVE_RIGHT, TARGET
from rl.tf.environments.line_world.state import LineWorldState


class LineWorldModel(object):

    def __init__(self, num_positions=10):
        self.num_positions = num_positions

    def move_left(self, position):
        return tf.maximum(position - 1, tf.Variable(0, dtype=tf.int32))

    def move_right(self, position):
        return tf.minimum(position + 1, tf.Variable(self.num_positions - 1, dtype=tf.int32))

    def apply_actions_vectorized(self, positions):
        r0 = tf.maximum(positions-1, 0)
        r1 = tf.minimum(positions+1, 9)
        return tf.stack([r0, r1], axis=1)

    def apply_actions(self, position):
        np0 = self.move_left(position)
        np1 = self.move_right(position)
        return tf.stack([np0, np1])

    def apply_action(self, position, action):
        cond = tf.equal(action, MOVE_LEFT)
        new_state = tf.cond(cond, lambda: self.move_left(position), lambda: self.move_right(position))
        return new_state

    def are_states_terminal_vectorized(self, positions):
        return tf.equal(positions, TARGET)

    def are_states_terminal(self, positions):
        return tf.map_fn(self.is_terminal, positions, dtype=tf.bool)

    def is_terminal(self, position):
        return tf.equal(position, TARGET)
    