import numpy as np
import logging

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


class LearnerMixin(object):

    def calculate_action_target(self, reward, next_state_action_values):
        raise NotImplemented()


class QLearnerMixin(LearnerMixin):

    def calculate_action_target(self, reward, next_state_action_values):
        return reward + self.gamma * next_state_action_values.max()


class SarsaLearnerMixin(LearnerMixin):

    def calculate_action_target(self, reward, next_state_action_values):
        pi = self.rl_system.policy.calculate_action_value_probabilities(next_state_action_values)
        # Note that using np.argmax(pi) is the greedy policy
        # TODO: This is sarse with greedy policy. Need to fix this so that
        # we can pick with greedy policy, softmax policy etc...
        next_state_action = np.argmax(pi)
        return reward + self.gamma * next_state_action_values[next_state_action]


class ExpecterSarsaLearnerMixin(LearnerMixin):
    '''
    target = R_{t+1} + \gamma * \E Q(S_{t+1}, A_{t+1})
           = R_{t+1} + \gamma * \sum_{a} \pi(a | S_{t+1} Q(S_{t+1}, a)

    '''

    def calculate_action_target(self, reward, next_state_action_values):
        pi = self.rl_system.policy.calculate_action_value_probabilities(next_state_action_values)
        print('pi is %s' % pi)
        expectation = np.dot(pi, next_state_action_values)
        return reward + self.gamma * expectation


class Learner:
    """

    """

    def __init__(self, rl_system):
        self.rl_system = rl_system

    def learn_episode(self, episode, **kwargs):
        state_array = episode.get_state_array()[:-1]
        targets = self.get_targets(episode)
        logging.info('Targets are:\n%s' % str(targets))
        self.rl_system.action_value_function.fit(state_array, targets, **kwargs)

    def get_targets(self, episode):
        """
        Generates the target values for training
        :param episode:
        :return: np.ndarray(len(episodes), num_actions
        """
        raise NotImplemented()


class RewardLearner(Learner):

    def get_targets(self, episode):
        targets = np.zeros((len(episode.states)-1, self.rl_system.num_actions))
        rewards = episode.get_reward_array()
        actions = episode.get_action_array()
        for i, (action, reward) in enumerate(zip(actions, rewards)):
            targets[i, action] = reward
        return targets


class NarrowLearner(Learner, LearnerMixin):

    def __init__(self, rl_system, gamma=1.0):
        self.rl_system = rl_system
        self.gamma = gamma

    def get_targets(self, episode):
        targets = np.zeros((len(episode.states) - 1, self.rl_system.num_actions))

        for i, state in enumerate(episode.states[:-1]):
            targets[i, :] = self.get_state_targets(state, episode.actions[i], episode.rewards[i])

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
        next_state_action_values = self.rl_system.action_value_function(next_state)
        targets = self.rl_system.action_value_function(state).ravel()
        targets[action] = self.calculate_action_target(reward, next_state_action_values)

        return targets


class SarsaLearner(NarrowLearner, SarsaLearnerMixin):
    '''
    target = R_{t+1} + \gamma * Q(S_{t+1}, A_{t+1})

    '''

    pass


class ExpectedSarsaLearner(NarrowLearner, ExpecterSarsaLearnerMixin):

    pass


class QLearner(NarrowLearner, QLearnerMixin):

    pass


class VectorLearner(LearnerMixin):

    def __init__(self, rl_system, gamma=1.0):
        self.rl_system = rl_system
        self.gamma = gamma

    def get_targets(self, episode):
        targets = np.zeros((len(episode.states), self.rl_system.num_actions))

        for i, state in enumerate(episode.states):
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
        return self.calculate_action_target(action_reward, next_state_action_values)


class VectorSarsaLearner(VectorLearner, SarsaLearnerMixin):

    pass


class VectorQLearner(VectorLearner, QLearnerMixin):
    """
    Wide Q learner.

    The basic Q Learner sets targets for a single action using

    targets = Q(s) # An ndarray with <num_actions> elements
    targets[action] = reward(action) + max(Q(next_state | action))          (***)

    i.e. The basic Q learner fits on <num_aciton> targets, but only 1 of the values is updated.

    The Wide Q Learner fits (***) for all actions. Thus making the fitting operation more efficient.

    """

    pass
