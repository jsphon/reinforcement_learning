import numpy as np
import tensorflow as tf

from rl.tf.lib.nested_loop_func import nested_loop
from rl.tf.lib.ordered_group_func import ordered_group


class NestedLoopTestCase(tf.test.TestCase):

    def setUp(self):
        self.c = tf.Variable(tf.zeros(100, tf.float32))
        self.i = tf.Variable(0)

    def append(self, value):
        def get_append_op():
            append_op = tf.scatter_update(self.c, self.i, value)
            print_op = tf.Print(append_op, [self.i, self.c[self.i]], 'Appending...')
            with tf.control_dependencies([append_op]):
                return print_op

        def get_inc_op():
            assign_op = tf.assign(self.i, self.i+1, use_locking=True)
            return assign_op

        return ordered_group(get_append_op, get_inc_op)

    def outer_loop_pre_func(self):
        op = self.append(-tf.to_float(self.i))
        return op

    def loop_body(self):
        op = self.append(tf.to_float(self.i))
        return op

    def test_nested_loop(self):

        nl = nested_loop(
            outer_steps=2,
            inner_steps=10,
            get_body_op=self.loop_body,
        )

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(nl)
            result = self.c[:self.i].eval()

        expected = np.arange(20)
        self.assertAllEqual(expected, result)

    def test_nested_loop_with_outer_loop_prefunc(self):

        nl = nested_loop(
            outer_steps=2,
            inner_steps=3,
            get_body_op=self.loop_body,
            get_pre_body_op=self.outer_loop_pre_func
        )

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(nl)
            result = self.c[:self.i].eval()

        expected = np.array([
            0, 1, 2, 3, -4, 5, 6, 7
        ])
        self.assertAllEqual(expected, result)