import logging
import textwrap

import numpy as np

from rl.lib.timer import Timer

np.set_printoptions(precision=2)
np.set_printoptions(linewidth=200)
np.set_printoptions(suppress=True)

logging.basicConfig(level=logging.DEBUG)


class LearningSystem(object):
    def __init__(self, learner, experience_generator):
        self.learner = learner
        self.experience_generator = experience_generator
        self.rl_system = learner.rl_system

    def learn_many_times(self, n):
        for i in range(n):
            self.learn_once(i)

    def learn_once(self, i=None):
        episodes = self.experience_generator.generate_episodes(num_episodes=1, max_len=20)
        average_reward = np.mean([sum(episode.rewards) for episode in episodes])

        with Timer('Learning'):
            self.learner.learn(episodes)

        print('=== AVERAGE REWARD %s' % average_reward)
        print('=== Value Function %s ===' % (i or ''))
        if hasattr(self.rl_system, 'get_value_grid'):
            print(self.rl_system.get_value_grid())

        if hasattr(self.rl_system, 'get_greedy_action_grid_string'):
            print('=== Greedy Actions %s ===' % (i or ''))
            actions = self.rl_system.get_greedy_action_grid_string()
            print(textwrap.indent(actions, ' '))
