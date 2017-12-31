
import tensorflow as tf

from rl.tf.lib.ordered_group import ordered_group


class OrderedGrouptestCase(tf.test.TestCase):

    def test_ordered_group1(self):
        c = tf.Variable(0)

        add_one = tf.assign(c, c+1)
        times_two = tf.assign(c, 2 * c)
        pow_three = tf.assign(c, tf.pow(c, 3))

        op = ordered_group(add_one, times_two, pow_three)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(op)
            result = c.eval()

        expected = ((0+1)*2)**3 # 8
        self.assertEqual(expected, result)

    def test_ordered_group2(self):
        c = tf.Variable(0)

        add_one = tf.assign(c, c+1)
        times_two = tf.assign(c, 2 * c)
        pow_three = tf.assign(c, tf.pow(c, 3))

        op = ordered_group(pow_three, times_two, add_one)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(op)
            result = c.eval()

        expected = ((0**3)*2)+1 # 3
        self.assertEqual(expected, result)