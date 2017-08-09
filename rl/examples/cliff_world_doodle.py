import logging

import numpy as np

from rl.core.experience import ExperienceGenerator, Episode, StatesList
from rl.core.learner import QLearner, SarsaLearner, ExpectedSarsaLearner,\
    VectorQLearner, VectorSarsaLearner
from rl.environments.cliff_world import CliffWorld, GridState
from rl.core.policy import SoftmaxPolicy
from rl.lib.timer import Timer
import textwrap

np.set_printoptions(precision=1)
np.set_printoptions(linewidth=200)
np.set_printoptions(suppress=True)

logging.basicConfig(level=logging.DEBUG)

grid_world = CliffWorld()
grid_world.policy.epsilon = 0.5

generator = ExperienceGenerator(grid_world)
episode = generator.generate_episode()

states = StatesList()
for i in range(grid_world.shape[0]):
    for j in range(grid_world.shape[1]):
        state = GridState(player=(i, j))
        states.append(state)

#learner = QLearner(grid_world)
#learner = SarsaLearner(grid_world)
#learner = VectorQLearner(grid_world)
learner = VectorSarsaLearner(grid_world)


def learn_once():

    learner.learn(states, epochs=10, verbose=0)

    print('=== Value Function ===')
    print(grid_world.get_value_grid())

    print('=== Greedy Actions ===')
    actions = grid_world.get_greedy_action_grid_string()
    print(textwrap.indent(actions, ' '))

for _ in range(1000):
    learn_once()

print( learner.get_target_array(StatesList([GridState((1, 11))])) )