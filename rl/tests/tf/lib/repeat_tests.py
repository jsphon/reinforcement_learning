

import tensorflow as tf

from rl.tf.lib import repeat


class RepeatTestCase(tf.test.TestCase):

    def test_repeat(self):
        c = tf.Variable(0)

        def make_op():
            op = tf.assign(c, c+1)
            op = tf.Print(op, [c], message='hello')
            return tf.group(op)

        repeat_op = repeat(make_op, 10)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            r0 = sess.run(repeat_op)
            print(r0)
            result = c.eval()

        self.assertEqual(10, result)
