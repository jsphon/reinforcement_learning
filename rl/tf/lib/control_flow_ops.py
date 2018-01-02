import tensorflow as tf



import tensorflow as tf


def for_loop(body, n):
    """
    Repeat the operation created by make_op n times.

    Args:
        make_op: a function that returns the operation we want to repeat
        n: the number times we want to repeat the operation

    Returns:
        An operation

    """
    i = tf.constant(0)

    def while_cond(counter):
        return tf.less(counter, n)

    def while_body(counter):
        op = body(i)
        with tf.control_dependencies([op]):
            return counter + 1

        return tf.while_loop(cond, body, [i])

    return tf.while_loop(while_cond, while_body, [i])



def ordered_group(*ops):
    """
    Create an op that groups multiple operations, which will be executed in order.

    Args:
        *make_ops: A list of functions that each return an operation. These operations
                    will be executed in the order that they appear
                    in this list

    Returns: An operation that executes all the operations in order.

    """
    if len(ops) == 1:
        return tf.identity(ops[0])
    elif len(ops) >= 2:
        op = ops[0]
        op = tf.identity(op)
        with tf.control_dependencies([op]):
            return ordered_group(*tf.identity_n(ops[1:]))
