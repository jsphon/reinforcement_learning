import logging

import numpy as np

from rl.core.experience import ExperienceGenerator
from rl.core.learner import QLearner, SarsaLearner, ExpectedSarsaLearner
from rl.environments.simple_grid_world import SimpleGridWorld
from rl.core.policy import SoftmaxPolicy

np.set_printoptions(precision=1)
np.set_printoptions(linewidth=200)
np.set_printoptions(suppress=True)

logging.basicConfig(level=logging.DEBUG)

grid_world = SimpleGridWorld()
generator = ExperienceGenerator(grid_world)
episode = generator.generate_episode()

learner = QLearner(grid_world)
learner = SarsaLearner(grid_world)

learner = ExpectedSarsaLearner(grid_world)
learner.rl_system.policy = SoftmaxPolicy(grid_world)

for i in range(500):
    episode = generator.generate_episode()
    learner.learn(episode, verbose=0)

    print('=== Value Function %i ===' % i)
    print(grid_world.get_value_grid())

print('=== Reward Function ===')
print(grid_world.get_reward_grid())

print('=== Value Function ===')
print(grid_world.get_value_grid())

print('=== Greedy Actions ===')
print(grid_world.get_greedy_action_grid_string())
