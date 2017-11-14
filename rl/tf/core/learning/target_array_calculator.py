import tensorflow as tf
import numpy as np


class ModelBasedTargetArrayCalculator(object):
    """
    Class for calculating the training target arrays

    Each element of a target array should correspond to the target for that element's action.
    """

    def __init__(self, rl_system, action_target_calculator):
        self.rl_system = rl_system
        self.action_target_calculator = action_target_calculator

    #
    # def get_target_matrix(self, states):
    #     """
    #     Get the training targets as an array
    #     Args:
    #         states:
    #
    #     Returns:
    #         np.ndarray: (len(experience), num_actions)
    #
    #     """
    #     targets_list = [self.get_state_targets(state) for state in states]
    #     targets_array = np.stack(targets_list)
    #
    #     return targets_array
    #
    # def get_state_targets(self, state):
    #     """
    #     Return the targets for the state
    #     :param state:
    #     :param action:
    #     :param reward:
    #     :return: np.ndarray(num_actions)
    #     """
    #
    #     num_actions = self.rl_system.num_actions
    #     targets = np.empty(num_actions)
    #
    #     for action in range(num_actions):
    #         targets[action] = self.get_state_action_target(state, action)
    #
    #     return targets

    # TODO: Can we have a state action target calculator class?
    def get_state_action_target(self, state, action):
        """
        Return the target for the state/action pair
        :param state:
        :param action:
        :param reward:
        :return: np.ndarray(num_actions)
        """
        next_state = self.rl_system.model.apply_action(state, action)
        reward = self.rl_system.reward_function(state, action, next_state)

        predicate = self.rl_system.model.is_terminal(next_state)

        def when_terminal():
            return reward

        def when_non_terminal():
            next_state_action_values = self.rl_system.action_value_function(next_state)
            return self.action_target_calculator.calculate(reward, next_state_action_values)

        target = tf.cond(predicate, when_terminal, when_non_terminal)

        return target
