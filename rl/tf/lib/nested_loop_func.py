import tensorflow as tf

from rl.tf.lib.ordered_group_func import ordered_group
from . import repeat

def nested_loop(
        outer_steps,
        inner_steps,
        body,
        outer_loop_pre_func=None,
    ):
    """
    Created a (single) nested loop operation.

    This is similar to the following Python code:

    for i in range(outer_steps):
        outer_loop_pre_func()
        for j in range(inner_steps):
            inner_loop_func()

    The original purpose of this function was for training loops, where we want to
    do some variable assignment every few steps

    Args:


    Returns: An operation.

    """

    def get_inner_loop():

        def _get():
            return repeat(body, inner_steps)

        if outer_loop_pre_func:
            return ordered_group(outer_loop_pre_func, _get)

        else:
            return _get()

    outer_loop = repeat(get_inner_loop, outer_steps)

    return outer_loop
