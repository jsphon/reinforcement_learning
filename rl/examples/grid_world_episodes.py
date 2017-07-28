import logging

import numpy as np

from rl.core.experience import ExperienceGenerator
from rl.core.learner import ExpectedSarsaLearner
from rl.environments.grid_world import GridState, GridWorld

np.set_printoptions(precision=1)
np.set_printoptions(linewidth=200)
np.set_printoptions(suppress=True)

logging.basicConfig(level=logging.DEBUG)

initial_states = GridState.all()

grid_world = GridWorld()

generator = ExperienceGenerator(grid_world)

# learner = QLearner(grid_world)
# learner = RewardLearner(grid_world)
# learner = QLearner(grid_world)
learner = ExpectedSarsaLearner(grid_world)

from rl.core.policy import SoftmaxPolicy

learner.rl_system.policy = SoftmaxPolicy(grid_world)

print('=== Initial Value Function ===')
print(grid_world.get_value_grid())

learner.gamma = 1.0
for i in range(20):
    episodes = []
    print('Generating episodes')
    for _ in range(100):
        episode = generator.generate_episode()
        episodes.append(episode)
    print('fitting model')
    learner.learn_episodes(episodes, epochs=1, verbose=0)
    print('=== Value Function %s ===' % i)
    print(grid_world.get_value_grid())

print('=== Reward Function ===')
print(grid_world.get_reward_grid())

print('=== Value Function ===')
print(grid_world.get_value_grid())

print('=== Greedy Actions ===')
print(grid_world.get_greedy_action_grid_string())
