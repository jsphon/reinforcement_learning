import numpy as np

from rl.core.learning.action_target_calculator import \
    SarsaActionTargetCalculator, \
    ExpectedSarsaActionTargetCalculator, \
    QLearningActionTargetCalculator

from rl.core.state import IntExtState, StateList

#
# class ExperienceBasedTargetArrayCalculator(object):
#     """
#     Class for calculating the training target arrays
#
#     Each element of a target array should correspend to the target for that element's action.
#     """
#
#     def __init__(self, rl_system, action_target_calculator):
#         self.rl_system = rl_system
#         self.action_target_calculator = action_target_calculator
#
#     def get_target_array(self, experience):
#         raise NotImplemented()
#
#     def get_target(self, state, action, reward, next_state):
#         """
#         Return the targets for the state
#         :param state:
#         :param action:
#         :param reward:
#         :return: np.ndarray(num_actions)
#         """
#         # reward is a function of (state, action, next state)
#         #next_state = self.rl_system.model.apply_action(state, action)
#
#         if next_state.is_terminal:
#             target = reward
#         else:
#             next_state_action_values = self.rl_system.action_value_function(next_state)
#             target = self.action_target_calculator.calculate(reward, next_state_action_values)
#
#         return target
#
#
# class ScalarTargetArrayCalculator(ExperienceBasedTargetArrayCalculator):
#
#     def get_target_array(self, experience):
#         """
#         Return a 1d array of targets for each action in experience
#         Args:
#             experience:
#
#         Returns:
#             targets: 1-d array
#
#         """
#
#         targets = np.zeros(experience.get_training_length())
#         actions = experience.get_training_actions()
#         rewards = experience.get_training_rewards()
#         states = experience.get_training_states()
#         for i, state in enumerate(states):
#             action = actions[i]
#             reward = rewards[i]
#             target_value = self.get_target(state, action, reward)
#             targets[i] = target_value
#
#         return targets
#
#
# class SemiVectorizedTargetArrayCalculator(ExperienceBasedTargetArrayCalculator):
#     """
#     Only calculates the target for the sampled action
#     """
#
#     def get_target_array(self, experience):
#         """
#         Get the training targets as an array
#         Args:
#             experience:
#
#         Returns:
#             np.ndarray: (len(experience), num_actions)
#
#         """
#         targets = np.zeros((experience.get_training_length(), self.rl_system.num_actions))
#         actions = experience.get_training_actions()
#         rewards = experience.get_training_rewards()
#         states = experience.get_training_states()
#         for i, state in enumerate(states):
#             targets[i, :] = self.get_state_targets(state, actions[i], rewards[i])
#
#         return targets
#
#     def get_state_targets(self, state, action, reward):
#         """
#         Return the targets for the state
#         :param state:
#         :param action:
#         :param reward:
#         :return: np.ndarray(num_actions)
#         """
#         targets = self.rl_system.action_value_function(state).ravel()
#         targets[action] = self.get_target(state, action, reward)
#         return targets


class ModelBasedTargetArrayCalculator(object):
    """
    Class for calculating the training target arrays

    Each element of a target array should correspond to the target for that element's action.
    """

    def __init__(self, rl_system, action_target_calculator):
        self.rl_system = rl_system
        self.action_target_calculator = action_target_calculator

    def get_target_matrix(self, states):
        """
        Get the training targets as an array
        Args:
            states:

        Returns:
            np.ndarray: (len(experience), num_actions)

        """
        targets_list = [self.get_state_targets(state) for state in states]
        targets_array = np.stack(targets_list)

        return targets_array

    def get_state_targets(self, state):
        """
        Return the targets for the state
        :param state:
        :param action:
        :param reward:
        :return: np.ndarray(num_actions)
        """

        #fail here. the next line is wrong
        #num_actions = state.num_actions
        num_actions = self.rl_system.num_actions
        targets = np.empty(num_actions)

        for action in range(num_actions):
            targets[action] = self.get_state_action_target(state, action)

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
        reward = self.rl_system.reward_function(state, action, next_state)

        if self.rl_system.model.is_terminal(next_state):
            target = reward
        else:
            next_state_action_values = self.rl_system.action_value_function(next_state)
            target = self.action_target_calculator.calculate(reward, next_state_action_values)

        return target


class ModelBasedStateMachineTargetArrayCalculator(ModelBasedTargetArrayCalculator):

    def get_target_matrix(self, external_states):

        targets = []
        for internal_state in range(self.rl_system.num_internal_states):
            int_ext_states = [IntExtState(internal_state, external_state) for external_state in external_states]
            i_targets = [self.get_state_targets(x) for x in int_ext_states]
            i_targets = np.r_[i_targets]
            targets.append(i_targets)

        return targets

    def get_state_targets(self, state):

        internal_state = state.internal_state
        num_actions = self.rl_system.num_actions[internal_state]
        targets = np.empty(num_actions)

        for action in range(num_actions):
            targets[action] = self.get_state_action_target(state, action)

        return targets


def build_target_array_calculator(
        rl_system,
        discount_factor=1.0,
        learning_algo='qlearning',
        calculator_type='modelbased'
        ):

    if learning_algo=='qlearning':
        action_target_calculator = QLearningActionTargetCalculator(rl_system, discount_factor=discount_factor)
    elif learning_algo=='sarsa':
        action_target_calculator = SarsaActionTargetCalculator(rl_system, discount_factor=discount_factor)
    elif learning_algo=='expectedsarsa':
        action_target_calculator = ExpectedSarsaActionTargetCalculator(rl_system, discount_factor=discount_factor)

    if calculator_type=='modelbased':
        return ModelBasedTargetArrayCalculator(rl_system, action_target_calculator)
    elif calculator_type=='modelbasedstatemachine':
        return ModelBasedStateMachineTargetArrayCalculator(rl_system, action_target_calculator)



#
# def build_sarsa_target_array_calculator(rl_system, discount_factor=1.0):
#     action_target_calculator = SarsaActionTargetCalculator(rl_system, discount_factor=discount_factor)
#     return ScalarTargetArrayCalculator(rl_system, action_target_calculator)
#
#
# def build_q_learning_target_array_calculator(rl_system, discount_factor=1.0):
#     action_target_calculator = QLearningActionTargetCalculator(rl_system, discount_factor=discount_factor)
#     return ScalarTargetArrayCalculator(rl_system, action_target_calculator)
#
#
# def build_expected_sarsa_target_array_calculator(rl_system, discount_factor=1.0):
#     action_target_calculator = ExpectedSarsaActionTargetCalculator(rl_system, discount_factor=discount_factor)
#     return ScalarTargetArrayCalculator(rl_system, action_target_calculator)
#
#
# #def build_vectorized_sarsa_target_array_calculator(rl_system, discount_factor=1.0):
# #    action_target_calculator = SarsaActionTargetCalculator(rl_system, discount_factor=discount_factor)
# #    return VectorizedTargetArrayCalculator(rl_system, action_target_calculator)
#
#
# #def build_vectorized_q_learning_target_array_calculator(rl_system, discount_factor=1.0):
# #    action_target_calculator = QLearningActionTargetCalculator(rl_system, discount_factor=discount_factor)
# #    return VectorizedTargetArrayCalculator(rl_system, action_target_calculator)
#
#
# def build_vectorized_expected_sarsa_target_array_calculator(rl_system, discount_factor=1.0):
#     action_target_calculator = ExpectedSarsaActionTargetCalculator(rl_system, discount_factor=discount_factor)
#     return SemiVectorizedTargetArrayCalculator(rl_system, action_target_calculator)
#
#
# #def build_vectorized_state_machine_q_learning_target_array_calculator(rl_system, discount_factor=1.0):
# #    action_target_calculator = QLearningActionTargetCalculator(rl_system, discount_factor=discount_factor)
# #    return VectorizedStateMachineTargetArrayCalculator(rl_system, action_target_calculator)
