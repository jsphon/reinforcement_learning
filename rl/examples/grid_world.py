import numpy as np

np.set_printoptions(precision=1)
np.set_printoptions(linewidth=200)
np.set_printoptions(suppress=True)

'''

Steps

1 . Generate Experience

    Generate n Episodes

        for i in range(n):

            Generate Sample Episode

    Simple Generate Sample Episode

        Initialise State

        while not terminal state:
            choose action
            take action

1 b.

    Generate root nodes
    For each root node:
        generate target value





2. Initialise StateAction Function (experience)
 - convert episode states to vectors
 - get rewards
 - fit the 1 step rewards against the target values so that we get decent starting values

3 . Fit StateAction Function (experience)

    Get States Array
        i.e. make a big array of states x num experience
    Get Target Values
        i.e. given the learner, such as sarsa or q, generate target values
    Fit(States, Target Value)


'''

from rl.environments.grid_world import GridState, GridWorld
from rl.core.learner import ExpectedSarsaLearner
from rl.core.experience import ExperienceGenerator
import logging
logging.basicConfig(level=logging.DEBUG)

initial_states = GridState.all()

grid_world = GridWorld()

generator = ExperienceGenerator(grid_world)
episode = generator.generate_episode()

learner = ExpectedSarsaLearner(grid_world)
learner.learn_episode(episode)

for _ in range(1000):
    episode = generator.generate_episode()
    learner.learn_episode(episode)

print('=== Reward Function ===')
print(grid_world.get_reward_grid())

print('=== Value Function ===')
print(grid_world.get_value_grid())

print('=== Greedy Actions ===')
print(grid_world.get_action_grid_string())