import numpy as np
import logging


class Learner:
    """

    """

    def __init__(self, rl_system):
        self.rl_system = rl_system

    def learn_episode(self, episode, **kwargs):
        state_array = episode.get_state_array()
        targets = self.get_targets(episode)
        logging.info('Targets are:\n%s' % str(targets))
        self.rl_system.action_value_function.fit(state_array, targets, **kwargs)

    def get_targets(self, episode):
        targets = np.zeros((len(episode.states), self.rl_system.num_actions))
        rewards = episode.get_reward_array()
        actions = episode.get_action_array()
        for i, (action, reward) in enumerate(zip(actions, rewards)):
            targets[i, action] = reward
        return targets
