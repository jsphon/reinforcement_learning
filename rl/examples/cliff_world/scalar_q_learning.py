import textwrap

from rl.core.experience import ExperienceGenerator
from rl.core.learning.learner import build_learner
from rl.core.learning.learning_system import LearningSystem
from rl.environments.cliff_world import CliffWorld
from rl.lib.timer import Timer

grid_world = CliffWorld()
grid_world.policy.epsilon = 0.1
grid_world.action_value_function.learning_rate = 0.05

generator = ExperienceGenerator(grid_world)
learner = build_learner(grid_world)

learning_system = LearningSystem(learner, generator)
with Timer('training') as t:
    learning_system.learn_many_times(1000)

print('=== Value Function ===')
print(grid_world.get_value_grid())

print('=== Greedy Actions ===')
# print(grid_world.get_greedy_action_grid_string())
greedy_actions = grid_world.get_greedy_action_grid_string()
print(textwrap.indent(greedy_actions, ' '))
