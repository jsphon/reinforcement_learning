from rl.core.learning.learner import build_learner
from rl.core.state import StateList
from rl.environments.line_world.rl_system import AntWorld
from rl.environments.line_world.rl_system import calculate_greedy_actions
from rl.environments.line_world.state import AntState
from rl.environments.line_world.constants import HOME_POSITION, FOOD_POSITION, FINDING_FOOD, FINDING_HOME

world = AntWorld()
learner = build_learner(world, calculator_type='modelbasedstatemachine')

states = [AntState(position=position) for position in range(10)]
states_list = StateList(states)

initial_greedy_actions = calculate_greedy_actions(world)

for _ in range(500):
    learner.learn(states_list, epochs=1)

greedy_actions = calculate_greedy_actions(world)

print('Initial Greedy Actions (should be random):')
print(initial_greedy_actions)

print('Optimised Greedy Actions (should point at home(%s) and food(%s) positions):' % (HOME_POSITION, FOOD_POSITION))
print(greedy_actions)

action_values = world.calculate_action_values()

print('Home is at position %s' % HOME_POSITION)
print('Action values for FINDING_HOME state:')
print(action_values[FINDING_HOME])

print('Food is at position %s' % FOOD_POSITION)
print('Action values for FINDING_FOOD state:')
print(action_values[FINDING_FOOD])
