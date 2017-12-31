
import tensorflow as tf

from rl.tf.lib.ordered_group_func import ordered_group


class OrderedGrouptestCase(tf.test.TestCase):

    def setUp(self):

        self.c = tf.Variable(0)

    def add_one(self):
        r = tf.assign(self.c, self.c+1)
        r = tf.Print(r, [self.c], 'Adding one')
        return r

    def times_two(self):
        r = tf.assign(self.c, 2 * self.c)
        r = tf.Print(r, [self.c], 'times two')
        return r

    def pow_three(self):
        r = tf.assign(self.c, tf.pow(self.c, 3))
        r = tf.Print(r, [self.c], 'pow three')
        return r

    def test_ordered_group1(self):

        op = ordered_group(self.add_one, self.times_two, self.pow_three)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(op)
            result = self.c.eval()

        expected = ((0+1)*2)**3 # 8
        self.assertEqual(expected, result)

    def test_ordered_group2(self):

        op = ordered_group(self.pow_three, self.times_two, self.add_one)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(op)
            result = self.c.eval()

        expected = ((0**3)*2)+1 # 3
        self.assertEqual(expected, result)