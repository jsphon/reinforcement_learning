import logging

import numpy as np

from rl.core.experience import ExperienceGenerator, StatesList
from rl.core.learner import QLearner, SarsaLearner, ExpectedSarsaLearner
from rl.environments.cliff_world import CliffWorld, GridState
from rl.core.policy import SoftmaxPolicy
from rl.lib.timer import Timer
import textwrap

np.set_printoptions(precision=2)
np.set_printoptions(linewidth=200)
np.set_printoptions(suppress=True)

logging.basicConfig(level=logging.DEBUG)

grid_world = CliffWorld()
grid_world.policy.epsilon=0.001
grid_world.action_value_function.learning_rate=0.05
generator = ExperienceGenerator(grid_world)

learner = QLearner(grid_world)
learner = SarsaLearner(grid_world)

#
# learner = ExpectedSarsaLearner(grid_world)
# learner.rl_system.policy = SoftmaxPolicy(grid_world)


def learn_many_times(n):

    for i in range(n):
        learn_once(i)


def learn_once(i=None):
    episodes = generator.generate_episodes(num_episodes=1, max_len=20)
    average_reward = np.mean([sum(episode.rewards) for episode in episodes])

    with Timer('Learning'):
        #learner.learn(episodes, verbose=1, epochs=1)
        learner.learn_1d(episodes)

    print('=== AVERAGE REWARD %s' % average_reward)
    print('=== Value Function %s ===' % (i or ''))
    print(grid_world.get_value_grid())

    print('=== Greedy Actions %s ===' % (i or ''))
    actions = grid_world.get_greedy_action_grid_string()
    print(textwrap.indent(actions, ' '))

with Timer('training') as t:

    learn_many_times(1)


print('=== Value Function ===')
print(grid_world.get_value_grid())

print('=== Greedy Actions ===')
#print(grid_world.get_greedy_action_grid_string())
actions = grid_world.get_greedy_action_grid_string()
print(textwrap.indent(actions, ' '))