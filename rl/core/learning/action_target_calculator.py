import numpy as np


class ActionTargetCalculator(object):
    """
    For calculating the target for a single action
    """
    def __init__(self, rl_system, discount_factor=1.0):
        self.rl_system = rl_system
        self.discount_factor = discount_factor

    def calculate(self, reward, next_state_action_values):
        raise NotImplemented()


class QLearningActionTargetCalculator(ActionTargetCalculator):
    def calculate(self, reward, next_state_action_values):
        return reward + self.discount_factor * next_state_action_values.max()


class SarsaActionTargetCalculator(ActionTargetCalculator):
    def calculate(self, reward, next_state_action_values):
        pi = self.rl_system.policy.calculate_action_value_probabilities(next_state_action_values)
        action = np.random.choice(len(pi), p=pi)
        return reward + self.discount_factor * next_state_action_values[action]


class ExpectedSarsaActionTargetCalculator(ActionTargetCalculator):
    '''
    target = R_{t+1} + \gamma * \E Q(S_{t+1}, A_{t+1})
           = R_{t+1} + \gamma * \sum_{a} \pi(a | S_{t+1} Q(S_{t+1}, a)

    where \gamma is the discount factor

    '''

    def calculate(self, reward, next_state_action_values):
        pi = self.rl_system.policy.calculate_action_value_probabilities(next_state_action_values)
        num_actions = len(next_state_action_values)
        pi = pi.reshape((1, num_actions))
        next_state_action_values = next_state_action_values.reshape((num_actions, 1))
        expectation = np.dot(pi, next_state_action_values)
        return reward + self.discount_factor * expectation
