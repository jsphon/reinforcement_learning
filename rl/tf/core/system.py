import tensorflow as tf


class System(object):

    def choose_action(self, state):
        probabilities = self.calculate_state_probabilities(state)
        action = tf.multinomial(probabilities, num_samples=1)
        return action[0][0]

