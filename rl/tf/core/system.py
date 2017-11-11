import tensorflow as tf


class System(object):

    def __init__(self):
        self.action_value_function = None
        self.policy = None
        self.model = None

        # If the model's state representation is different
        # to the representation required by the action value
        # calculator, this function will transform it.
        # E.G. in a grid world it is better for the model to represent
        # the position as coordinates, but the action value function
        # might prefer to have the representation as a 1-hot array
        self.to_action_value_function_input = None

    def choose_action(self, state):
        if self.to_action_value_function_input:
            state = self.to_action_value_function_input(state)
        probabilities = self.calculate_state_probabilities(state)
        action = tf.multinomial(probabilities, num_samples=1)
        return action[0][0]

    def calculate_action_value_probabilities(self, state):
        action_values = self.action_value_function(state)
        return self.policy.calculate_action_value_probabilities(action_values)


