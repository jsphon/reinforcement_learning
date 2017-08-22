import numpy as np

from rl.core.learning.action_target_calculator import \
    SarsaActionTargetCalculator, \
    ExpectedSarsaActionTargetCalculator, \
    QLearningTargetCalculator


class TargetArrayCalculator(object):
    """
    Class for calculating the training target arrays

    Each element of a target array should correspend to the target for that element's action.
    """

    def __init__(self, rl_system, action_target_calculator):
        self.rl_system = rl_system
        self.action_target_calculator = action_target_calculator

    def get_target_array(self):
        raise NotImplemented()


class ScalarTargetArrayCalculator(TargetArrayCalculator):
    def get_target_array(self, experience):
        """
        Return a 1d array of targets for each action in experience
        Args:
            experience:

        Returns:

        """

        targets = np.zeros(experience.get_training_length())
        actions = experience.get_training_actions()
        rewards = experience.get_training_rewards()
        states = experience.get_training_states()
        for i, state in enumerate(states):
            action = actions[i]
            reward = rewards[i]
            target_value = self.get_target(state, action, reward)
            targets[i] = target_value

        return targets

    def get_target(self, state, action, reward):
        """
        Return the targets for the state
        :param state:
        :param action:
        :param reward:
        :return: np.ndarray(num_actions)
        """
        next_state = self.rl_system.model.apply_action(state, action)

        if next_state.is_terminal:
            target = reward
        else:
            next_state_action_values = self.rl_system.action_value_function(next_state)
            target = self.action_target_calculator.calculate(reward, next_state_action_values)

        return target


class VectorizedTargetArrayCalculator(TargetArrayCalculator):
    def get_target(self, state, action, reward):
        """
        Return the targets for the state
        :param state:
        :param action:
        :param reward:
        :return: np.ndarray(num_actions)
        """
        next_state = self.rl_system.model.apply_action(state, action)

        if next_state.is_terminal:
            target = reward
        else:
            next_state_action_values = self.rl_system.action_value_function(next_state)
            target = self.action_target_calculator.calculate(reward, next_state_action_values)

        return target

    def get_target_array(self, experience):
        """
        Get the training targets as an array
        Args:
            experience:

        Returns:
            np.ndarray: (len(experience), num_actions)

        """
        targets = np.zeros((experience.get_training_length(), self.rl_system.num_actions))
        actions = experience.get_training_actions()
        rewards = experience.get_training_rewards()
        states = experience.get_training_states()
        for i, state in enumerate(states):
            targets[i, :] = self.get_state_targets(state, actions[i], rewards[i])

        return targets

    def get_state_targets(self, state, action, reward):
        """
        Return the targets for the state
        :param state:
        :param action:
        :param reward:
        :return: np.ndarray(num_actions)
        """
        next_state = self.rl_system.model.apply_action(state, action)
        targets = self.rl_system.action_value_function(state).ravel()

        if next_state.is_terminal:
            targets[action] = reward
        else:
            next_state_action_values = self.rl_system.action_value_function(next_state)
            targets[action] = self.action_target_calculator.calculate(reward, next_state_action_values)

        return targets


def build_sarsa_target_array_calculator(rl_system, discount_factor = 1.0):
    action_target_calculator = SarsaActionTargetCalculator(rl_system, discount_factor=discount_factor)
    return ScalarTargetArrayCalculator(rl_system, action_target_calculator)


def build_q_learning_target_array_calculator(rl_system, discount_factor=1.0):
    action_target_calculator = QLearningTargetCalculator(rl_system, discount_factor=discount_factor)
    return ScalarTargetArrayCalculator(rl_system, action_target_calculator)


def build_expected_sarsa_target_array_calculator(rl_system, discount_factor=1.0):
    action_target_calculator = ExpectedSarsaActionTargetCalculator(rl_system, discount_factor=discount_factor)
    return ScalarTargetArrayCalculator(rl_system, action_target_calculator)


def build_vectorized_sarsa_target_array_calculator(rl_system, discount_factor = 1.0):
    action_target_calculator = SarsaActionTargetCalculator(rl_system, discount_factor=discount_factor)
    return VectorizedTargetArrayCalculator(rl_system, action_target_calculator)


def build_vectorized_q_learning_target_array_calculator(rl_system, discount_factor=1.0):
    action_target_calculator = QLearningTargetCalculator(rl_system, discount_factor=discount_factor)
    return VectorizedTargetArrayCalculator(rl_system, action_target_calculator)


def build_vectorized_expected_sarsa_target_array_calculator(rl_system, discount_factor=1.0):
    action_target_calculator = ExpectedSarsaActionTargetCalculator(rl_system, discount_factor=discount_factor)
    return VectorizedTargetArrayCalculator(rl_system, action_target_calculator)
