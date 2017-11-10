import tensorflow as tf


class System(object):

    def __init__(self):
        self.action_value_function = None

    def choose_action(self, state):
        probabilities = self.calculate_state_probabilities(state)
        action = tf.multinomial(probabilities, num_samples=1)
        return action[0][0]

    def calculate_action_value_probabilities(self, state):
        action_values = self.action_value_function(state)
        return self.policy.calculate_action_value_probabilities(action_values)


