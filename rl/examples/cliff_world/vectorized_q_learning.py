import textwrap

from rl.core.learning.learner import build_learner
from rl.environments.cliff_world import CliffWorld, GridState
from rl.lib.timer import Timer

grid_world = CliffWorld()
grid_world.policy.epsilon = 0.1
grid_world.action_value_function.learning_rate = 0.05

states = GridState.all()

learner = build_learner(grid_world)

with Timer('training') as t:
    for i in range(100):
        print('Epoch %s' % i)
        learner.learn(states)

print('=== Value Function ===')
print(grid_world.get_value_grid())

print('=== Greedy Actions ===')
greedy_actions = grid_world.get_greedy_action_grid_string()
print(textwrap.indent(greedy_actions, ' '))
