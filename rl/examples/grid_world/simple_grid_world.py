import logging

import numpy as np

from rl.core.experience import ExperienceGenerator
from rl.core.learning.learner import build_q_learner
from rl.environments.grid_world.simple_grid_world import SimpleGridWorld
from rl.lib.timer import Timer

np.set_printoptions(precision=1)
np.set_printoptions(linewidth=200)
np.set_printoptions(suppress=True)

logging.basicConfig(level=logging.DEBUG)


grid_world = SimpleGridWorld()

generator = ExperienceGenerator(grid_world)

learner = build_q_learner(grid_world)

print('=== Initial Value Function ===')
print(grid_world.get_value_grid())

learner.gamma = 1.0
for i in range(20):
    with Timer('Generating Episodes'):
        episodes = generator.generate_episodes(100, max_len=10)
    with Timer('Learning Model'):
        learner.learn(episodes)

print('=== Reward Function ===')
print(grid_world.get_reward_grid())

print('=== Value Function ===')
print(grid_world.get_value_grid())

print('=== Greedy Actions ===')
print(grid_world.get_greedy_action_grid_string())
