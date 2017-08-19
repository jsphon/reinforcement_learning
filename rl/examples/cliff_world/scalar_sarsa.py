import logging

import numpy as np

from rl.core.experience import ExperienceGenerator
from rl.core.learner import build_sarsa_learner
from rl.environments.cliff_world import CliffWorld
from rl.lib.timer import Timer
import textwrap

np.set_printoptions(precision=2)
np.set_printoptions(linewidth=200)
np.set_printoptions(suppress=True)

logging.basicConfig(level=logging.DEBUG)

grid_world = CliffWorld()
grid_world.policy.epsilon = 0.1
grid_world.action_value_function.learning_rate = 0.05
generator = ExperienceGenerator(grid_world)

learner = build_sarsa_learner(grid_world)


def learn_many_times(n):
    for i in range(n):
        learn_once(i)


def learn_once(i=None):
    episodes = generator.generate_episodes(num_episodes=1, max_len=20)
    average_reward = np.mean([sum(episode.rewards) for episode in episodes])

    with Timer('Learning'):
        learner.learn(episodes)

    print('=== AVERAGE REWARD %s' % average_reward)
    print('=== Value Function %s ===' % (i or ''))
    print(grid_world.get_value_grid())

    print('=== Greedy Actions %s ===' % (i or ''))
    actions = grid_world.get_greedy_action_grid_string()
    print(textwrap.indent(actions, ' '))


with Timer('training') as t:
    learn_many_times(100)

print('=== Value Function ===')
print(grid_world.get_value_grid())

print('=== Greedy Actions ===')
# print(grid_world.get_greedy_action_grid_string())
greedy_actions = grid_world.get_greedy_action_grid_string()
print(textwrap.indent(greedy_actions, ' '))
