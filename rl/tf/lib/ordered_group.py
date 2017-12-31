import tensorflow as tf


def ordered_group(*ops):
    """
    Create an op that groups multiple operations, which will be executed in order.

    Args:
        *ops: A list of operations. Operations will be executed in the order that they appear
            in this lost

    Returns: An operation that executes all its inputs in order.

    """
    if len(ops) == 1:
        return ops[0]
    elif len(ops) == 2:
        with tf.control_dependencies(ops[:1]):
            return tf.identity(ops[-1])
    else:
        with tf.control_dependencies(ops[:1]):
            return ordered_group(*ops[1:])
