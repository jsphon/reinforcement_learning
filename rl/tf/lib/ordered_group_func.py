import tensorflow as tf


def ordered_group(*make_ops):
    """
    Create an op that groups multiple operations, which will be executed in order.

    Args:
        *make_ops: A list of functions that each return an operation. These operations
                    will be executed in the order that they appear
                    in this list

    Returns: An operation that executes all the operations in order.

    """
    if len(make_ops) == 1:
        return make_ops[0]()
    elif len(make_ops) == 2:
        op = make_ops[0]()
        with tf.control_dependencies([op]):
            return make_ops[-1]()
    else:
        op = make_ops[0]()
        with tf.control_dependencies([op]):
            return ordered_group(*make_ops[1:])
