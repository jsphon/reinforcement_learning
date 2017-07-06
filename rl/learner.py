import numpy as np
import logging


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


class QLearner(Learner):

    def __init__(self, rl_system, gamma=1.0):
        self.rl_system = rl_system
        self.gamma = gamma

    def get_targets(self, episode):
        targets = np.zeros((len(episode.states)-1, self.rl_system.num_actions))
        rewards = episode.get_reward_array()

        for i, state in enumerate(episode.states[:-1]):
            action = episode.actions[i]
            next_state = self.rl_system.model.apply_action(state, action)

            next_state_vector = next_state.as_array().reshape((1, -1))
            next_state_action_values = self.rl_system.action_value_function(next_state_vector)

            state_vector = state.as_array().reshape(1, -1)
            targets[i, :] = self.rl_system.action_value_function(state_vector)
            targets[i, action] = rewards[i] + self.gamma * next_state_action_values.max()

        return targets


class WQLearner(Learner):

    def __init__(self, rl_system, gamma=1.0):
        self.rl_system = rl_system
        self.gamma = gamma

    def get_targets(self, episode):
        targets = np.zeros((len(episode.states), self.rl_system.num_actions))
        rewards = episode.get_reward_array()

        for i, state in enumerate(episode.states):
            action = episode.actions[i]

            next_state_action_values = np.ndarray((self.rl_system.num_actions, self.rl_system.num_actions))

            for action in range(self.rl_system.num_actions):
                next_state = self.rl_system.model.apply_action(state, action)
                next_state_vector = next_state.as_vector().reshape((1, -1))
                next_state_action_values[action, :] = self.rl_system.action_value_function(next_state_vector)

            #CONTINUE TOOO TIRED

            next_state_vector = next_state.as_vector().reshape((1, -1))
            next_state_action_values = self.rl_system.action_value_function(next_state_vector)

            state_vector = state.as_vector().reshape(1, -1)
            targets[i, :] = self.rl_system.action_value_function(state_vector)
            targets[i, action] = rewards[i] + self.gamma * next_state_action_values.max()

        return targets
