import tensorflow as tf
import numpy as np

from rl.tests.tf.utils import evaluate_tensor

class ModelBasedTargetArrayCalculator(object):
    """
    Class for calculating the training target arrays

    Each element of a target array should correspond to the target for that element's action.
    """

    def __init__(self, rl_system, action_target_calculator):
        self.rl_system = rl_system
        self.action_target_calculator = action_target_calculator

    def get_states_targets(self, states):
        """
        Get the targets for a of states many
        Args:
            states: Tensor of shape (num_states, state_size)

        Returns:
            Tensor of shape (num_states, num_actions)
        """

        result = tf.map_fn(self.get_state_targets, states, dtype=tf.float32)
        return result

    def get_state_targets(self, state):
        """
        Get all targets for a single state, so the result will be a tensor
        of shape (1, num_actions)

        Args:
            state:

        Returns:

        """

        # 1 x num_actions
        next_states = self.rl_system.model.apply_actions(state)
        rewards = self.rl_system.reward_function.state_rewards(state, next_states)
        is_terminal = self.rl_system.model.are_states_terminal(next_states)

        # num_actions x num_actions
        next_states_action_values = self.rl_system.action_value_function.vectorized(next_states)
        next_state_targets = self.action_target_calculator.vectorized(rewards, next_states_action_values)

        # 1 x num_actions
        targets = tf.where(is_terminal, rewards, next_state_targets)
        return targets

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
        reward = self.rl_system.reward_function.action_reward(state, action, next_state)

        print(evaluate_tensor(reward))

        predicate = self.rl_system.model.is_terminal(next_state)

        def when_terminal():
            return reward

        def when_non_terminal():
            next_state_action_values = self.rl_system.action_value_function.calculate(next_state)
            return self.action_target_calculator.calculate(reward, next_state_action_values)

        target = tf.cond(predicate, when_terminal, when_non_terminal)

        return target
