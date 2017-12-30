
import tensorflow as tf


def repeat(make_op, n):
    """
    Repeat the operation created by make_op n times.

    Args:
        make_op: a function that returns the operation we want to repeat
        n: the number times we want to repeat the operation

    Returns:
        An operation

    """
    i = tf.constant(0)

    def cond(counter):
        return tf.less(counter, n)

    def body(counter):
        op = make_op()
        with tf.control_dependencies([op]):
            return counter + 1

        return tf.while_loop(cond, body, [i])

    return tf.while_loop(cond, body, [i])
