
import numpy as np
import tensorflow as tf

from rl.tf.lib.control_flow_ops import ordered_group, for_loop, nested_for_loop


class NestedForLoopTestCase(tf.test.TestCase):

    def setUp(self):
        self.c = tf.Variable(tf.zeros(20, tf.float32))
        self.i = tf.Variable(0)

    def append(self, value):
        def get_append_op():
            return tf.scatter_update(self.c, self.i, value)

        def get_inc_op():
            assign_op = tf.assign(self.i, self.i+1, use_locking=True)
            return assign_op

        def get_print_op():
            return tf.Print(tf.Variable(0), [self.i, self.c[0], self.c[1], self.c[2], self.c[3], self.c[4]], 'Appending...')

        return ordered_group(get_append_op, get_inc_op, get_print_op)

    def outer_body(self, i):
        op = self.append(-tf.to_float(self.i))
        return op

    def loop_body(self, j):
        op = self.append(tf.to_float(self.i))
        op = tf.Print(op, [j], 'loop body')
        return op

    def test_nested_for_loop(self):

        nl = nested_for_loop(
            outer_steps=2,
            inner_steps=10,
            inner_body=self.loop_body,
        )

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(nl)
            result = self.c[:self.i].eval()

        expected = np.arange(20)
        self.assertAllEqual(expected, result)

    def test_nested_loop_with_outer_body(self):

        nl = nested_for_loop(
            outer_steps=2,
            inner_steps=3,
            inner_body=self.loop_body,
            outer_body=self.outer_body
        )

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(nl)
            result = self.c[:self.i].eval()

        expected = np.array([
            0, 1, 2, 3, -4, 5, 6, 7
        ])
        self.assertAllEqual(expected, result)


class ForLoopTestcase(tf.test.TestCase):

    def test_for_loop(self):
        c = tf.Variable(0)

        def body(i):
            op = tf.assign(c, c + 1)
            op = tf.Print(op, [i, c], message='hello')
            return tf.group(op)

        repeat_op = for_loop(body, 10)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            r0 = sess.run(repeat_op)
            print(r0)
            result = c.eval()

        self.assertEqual(10, result)

#
# class OrderedGrouptestCase(tf.test.TestCase):
#
#     def setUp(self):
#
#         self.c = tf.Variable(0)
#
#     def add_one(self):
#         r = tf.assign(self.c, self.c+1)
#         r = tf.Print(r, [self.c], 'Adding one')
#         return r
#
#     def times_two(self):
#         r = tf.assign(self.c, 2 * self.c)
#         r = tf.Print(r, [self.c], 'times two')
#         return r
#
#     def pow_three(self):
#         r = tf.assign(self.c, tf.pow(self.c, 3))
#         r = tf.Print(r, [self.c], 'pow three')
#         return r
#
#     def test_ordered_group_add_one_times_two(self):
#
#         add_one = self.add_one()
#         times_two = self.times_two()
#
#         op = ordered_group(add_one, times_two)
#
#         with self.test_session() as sess:
#             sess.run(tf.global_variables_initializer())
#             sess.run(op)
#             result = self.c.eval()
#
#         expected = (0+1)*2 # 2
#         self.assertEqual(expected, result)
#
#     def test_ordered_group_add_one_times_two_pow_three(self):
#
#         add_one = self.add_one()
#         times_two = self.times_two()
#         pow_three = self.pow_three()
#
#         op = ordered_group(add_one, times_two, pow_three)
#
#         with self.test_session() as sess:
#             sess.run(tf.global_variables_initializer())
#             sess.run(op)
#             result = self.c.eval()
#
#         expected = ((0+1)*2)**3 # 8
#         self.assertEqual(expected, result)
#
#     def test_ordered_group2(self):
#
#         op = ordered_group(self.pow_three, self.times_two, self.add_one)
#
#         with self.test_session() as sess:
#             sess.run(tf.global_variables_initializer())
#             sess.run(op)
#             result = self.c.eval()
#
#         expected = ((0**3)*2)+1 # 3
#         self.assertEqual(expected, result)