import tensorflow as tf

def generate_samples(t, num_samples=1000):
    """
    Generate samples from the tensor t
    Args:
        t:
        num_samples:

    Returns:

    """

    i = tf.Variable(0)
    result = tf.TensorArray(t().dtype, num_samples)

    def cond(b_i, result):
        return tf.less(b_i, num_samples)

    def body(b_i, b_result):
        b_result = b_result.write(b_i, t())
        return b_i + 1, b_result

    i, result = tf.while_loop(cond, body, (i, result))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        return sess.run(result.stack())


def evaluate_tensor(t):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        return sess.run(t)
