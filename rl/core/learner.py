import numpy as np

"""
NOTES

The Q learner's get_state_targets function need input (state, action, reward)
Sarsa will be the same

The Wide Q Learner only needs state, as it peeks into ALL next actions

Next:

Implement
 - Vectorized Expected Sarsa

 - n-step Sarsa
 - n-step Expected Sarsa
 - n-step tree backup

"""


class ActionTargetCalculator(object):
    def __init__(self, rl_system, discount_factor=1.0):
        self.rl_system = rl_system
        self.discount_factor = discount_factor

    def calculate(self, reward, next_state_action_values):
        raise NotImplemented()


class QLearnerActionTargetCalculator(ActionTargetCalculator):
    def calculate(self, reward, next_state_action_values):
        return reward + self.discount_factor * next_state_action_values.max()


class SarsaActionTargetCalculator(ActionTargetCalculator):
    def calculate(self, reward, next_state_action_values):
        pi = self.rl_system.policy.calculate_action_value_probabilities(next_state_action_values)
        action = np.random.choice(len(pi), p=pi)
        return reward + self.discount_factor * next_state_action_values[action]


class ExpecterSarsaActionTargetCalculator(ActionTargetCalculator):
    '''
    target = R_{t+1} + \gamma * \E Q(S_{t+1}, A_{t+1})
           = R_{t+1} + \gamma * \sum_{a} \pi(a | S_{t+1} Q(S_{t+1}, a)

    where \gamma is the discount factor

    '''

    def calculate(self, reward, next_state_action_values):
        pi = self.rl_system.policy.calculate_action_value_probabilities(next_state_action_values)
        num_actions = self.rl_system.num_actions
        pi = pi.reshape((1, num_actions))
        next_state_action_values = next_state_action_values.reshape((num_actions, 1))
        expectation = np.dot(pi, next_state_action_values)
        return reward + self.discount_factor * expectation


class Learner(object):
    """

    """

    def __init__(self, rl_system, action_target_calculator=None):
        self.rl_system = rl_system
        self.action_target_calculator = action_target_calculator

    def learn(self, experience, **kwargs):
        target_array = self.get_target_array(experience)
        self.rl_system.action_value_function.fit(
            experience.get_training_states(),
            target_array,
            **kwargs)

    def get_target_array(self, experience):
        """
        Generates the target values for training
        :param experience:
        :return: np.ndarray of shape (len(episodes), num_actions)
        """
        raise NotImplemented()


class ScalarLearner(Learner):
    """
    Calculates action targets for a single action at a time
    """

    def __init__(self, rl_system, action_target_calculator=None):
        self.rl_system = rl_system
        self.action_target_calculator = action_target_calculator

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


class SarsaLearner(ScalarLearner):
    '''
    target = R_{t+1} + \gamma * Q(S_{t+1}, A_{t+1})
    '''

    def __init__(self, rl_system, discount_factor=1.0):
        super(SarsaLearner, self).__init__(rl_system)
        self.action_target_calculator = SarsaActionTargetCalculator(rl_system, discount_factor)


class ExpectedSarsaLearner(ScalarLearner):
    def __init__(self, rl_system, discount_factor=1.0):
        super(ExpectedSarsaLearner, self).__init__(rl_system)
        self.action_target_calculator = ExpecterSarsaActionTargetCalculator(rl_system, discount_factor)


class QLearner(ScalarLearner):
    def __init__(self, rl_system, discount_factor=1.0):
        super(QLearner, self).__init__(rl_system)
        self.action_target_calculator = QLearnerActionTargetCalculator(rl_system, discount_factor)


class VectorLearner(Learner):
    """

        The basic Scalar Learner sets targets for a single action using

        targets = Q(s) # An ndarray with <num_actions> elements
        targets[action] = reward(action) + max(Q(next_state | action))          (***)

        i.e. The basic static learner fits on <num_action> targets, but only 1 of the values is updated.

        The VectorLearner fits (***) for all actions. Thus making the fitting operation more efficient.

        """

    def __init__(self, rl_system, gamma=1.0):
        self.rl_system = rl_system
        self.discount_factor = gamma

    def get_target_array(self, episode):
        targets = np.zeros((len(episode.states), self.rl_system.num_actions))

        for i, state in enumerate(episode.states):
            if state.is_terminal:
                targets[i, :] = 0
            else:
                targets[i, :] = self.get_state_targets(state)

        return targets

    def get_state_targets(self, state):
        """
        Return the targets for the state
        :param state:
        :return: np.ndarray(num_actions)
        """
        targets = np.ndarray(self.rl_system.num_actions)
        for action in range(self.rl_system.num_actions):
            targets[action] = self.get_state_action_target(state, action)
        return targets

    def get_state_action_target(self, state, action):
        next_state = self.rl_system.model.apply_action(state, action)
        action_reward = self.rl_system.reward_function(state, action, next_state)
        next_state_action_values = self.rl_system.action_value_function(next_state)
        return self.action_target_calculator.calculate(action_reward, next_state_action_values)


class VectorSarsaLearner(VectorLearner):

    def __init__(self, rl_system, discount_factor=1.0):
        super(VectorSarsaLearner, self).__init__(rl_system)
        self.action_target_calculator = SarsaActionTargetCalculator(rl_system, discount_factor)


class VectorQLearner(VectorLearner):

    def __init__(self, rl_system, discount_factor=1.0):
        super(VectorQLearner, self).__init__(rl_system)
        self.action_target_calculator = QLearnerActionTargetCalculator(rl_system, discount_factor)
