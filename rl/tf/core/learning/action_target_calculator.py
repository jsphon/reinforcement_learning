import tensorflow as tf


class ActionTargetCalculator(object):
    """
    For calculating the target for a single action
    """
    def __init__(self, rl_system, discount_factor=1.0):
        self.rl_system = rl_system
        if not isinstance(discount_factor, tf.Tensor):
            self.discount_factor = tf.Variable(discount_factor)
        else:
            self.discount_factor = discount_factor

    def calculate(self, reward, next_state_action_values):
        raise NotImplemented()


class QLearningActionTargetCalculator(ActionTargetCalculator):

    def calculate(self, reward, next_state_action_values):
        return reward + self.discount_factor * tf.reduce_max(next_state_action_values)

    #TODO Write the test for this:
    def vectorized_1d(self, rewards, next_state_action_values):
        """

        Args:
            rewards: tf.Tensor of shape (num_actions, 1)
            next_state_action_values: tf.Tensor of shape (num_actions, num_actions)

        Returns:

        """
        return rewards + self.discount_factor * tf.reduce_max(next_state_action_values, axis=0)


class SarsaActionTargetCalculator(ActionTargetCalculator):
    def calculate(self, reward, next_state_action_values):
        pi = self.rl_system.policy.calculate_action_value_probabilities(next_state_action_values)
        logits = tf.log(pi)
        action = tf.multinomial([logits], 1)
        next_state_action_value = next_state_action_values[action[0][0]]
        return reward + self.discount_factor * next_state_action_value


class ExpectedSarsaActionTargetCalculator(ActionTargetCalculator):
    '''
    target = R_{t+1} + \gamma * \E Q(S_{t+1}, A_{t+1})
           = R_{t+1} + \gamma * \sum_{a} \pi(a | S_{t+1} Q(S_{t+1}, a)

    where \gamma is the discount factor

    '''

    def calculate(self, reward, next_state_action_values):
        pi = self.rl_system.policy.calculate_action_value_probabilities(next_state_action_values)
        expectation = tf.tensordot(pi, next_state_action_values, 1)
        return reward + self.discount_factor * expectation
