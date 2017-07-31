import logging

import numpy as np

from rl.lib.timer import Timer
from rl.core.experience import ExperienceGenerator
from rl.core.learner import ExpectedSarsaLearner, QLearner, SarsaLearner
from rl.environments.simple_grid_world import SimpleGridState, SimpleGridWorld

np.set_printoptions(precision=1)
np.set_printoptions(linewidth=200)
np.set_printoptions(suppress=True)

logging.basicConfig(level=logging.DEBUG)


grid_world = SimpleGridWorld()

generator = ExperienceGenerator(grid_world)

learner = QLearner(grid_world)
learner = SarsaLearner(grid_world)
# learner = QLearner(grid_world)
# learner = ExpectedSarsaLearner(grid_world)
#
# from rl.core.policy import SoftmaxPolicy
#
# learner.rl_system.policy = SoftmaxPolicy(grid_world)

print('=== Initial Value Function ===')
print(grid_world.get_value_grid())

learner.gamma = 1.0
for i in range(20):
    with Timer('Generating Episodes'):
        episodes = generator.generate_episodes(100, max_len=10)
    with Timer('Learning Model'):
        learner.learn(episodes, epochs=1, verbose=0)

print('=== Reward Function ===')
print(grid_world.get_reward_grid())

print('=== Value Function ===')
print(grid_world.get_value_grid())

print('=== Greedy Actions ===')
print(grid_world.get_greedy_action_grid_string())
