import tensorflow as tf


class EpsilonGreedyPolicy(object):
    def __init__(self, epsilon=tf.Variable(0.1)):
        if isinstance(epsilon, float):
            self.epsilon = tf.Variable(epsilon)
        else:
            self.epsilon = epsilon

    def __str__(self):
        return '<EpsilonGreedyPolicy epsilon=%s>'%self.epsilon

    def calculate_action_value_probabilities(self, action_values):
        num_actions = tf.size(action_values, out_type=action_values.dtype)
        default_probability = self.epsilon / (num_actions-1.0)
        ones = tf.ones_like(action_values)
        t_probabilities = ones * default_probability
        best_action = tf.argmax(action_values, axis=0)
        t_probabilities = tf.concat([t_probabilities[:best_action], [1.0-self.epsilon], t_probabilities[best_action+1:]], axis=0)
        return t_probabilities


class SoftmaxPolicy(object):

    def calculate_action_value_probabilities(self, action_values):
        return tf.nn.softmax(action_values)