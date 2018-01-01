from . import ordered_group, repeat


def nested_loop(
        outer_steps,
        inner_steps,
        get_body_op,
        get_pre_body_op=None,
):
    """
    Created a (single) nested loop operation.

    This is similar to the following Python code:

    for i in range(outer_steps):
        get_pre_body_op()() # double parenthesis is intentional
        for j in range(inner_steps):
            get_body_op()() # double parenthesis is intentional

    The original purpose of this function was for training loops, where we want to
    do some variable assignment every few steps

    Args:


    Returns: An operation.

    """

    def get_inner_loop():

        if get_pre_body_op:
            return ordered_group(get_pre_body_op, _get_inner_loop)

        else:
            return _get_inner_loop()

    def _get_inner_loop():
        return repeat(get_body_op, inner_steps)

    outer_loop = repeat(get_inner_loop, outer_steps)

    return outer_loop
