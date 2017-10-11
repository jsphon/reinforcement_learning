import numpy as np

from rl.core import State
from rl.core.rl_system import StateMachineSystem
from rl.core.policy import Policy
from rl.core.reward_function import RewardFunction
from rl.core.value_function import ActionValueFunction
from rl.core.model import Model
from rl.core.state import StateMeta, IntExtState

# Agent has gone to a town, with a restaurant and bar
# Internal states represent whether or not the agent is hungry or thirsty
# (mutually exclusive in this example)
# The agent can eat at the restaurant, or drink at the bar.
# From either location they can go home, which is the terminal state.


# Internal States

HUNGRY = 0
SATISFIED = 1

# External States

class BaseStateMeta(object):
    def __init__(self):
        self.size = 3
        self.array_dtype = np.bool
        self.num_actions = 2


# LOCATIONS

AT_BAR = 0
AT_RESTAURANT = 1
AT_HOME = 2


# Actions

## When Hungry

EAT = 0

## When Satisfied

GO_HOME = 0

## Universal
GO_TO_RESTAURANT = 1
GO_TO_BAR = 2


class AgentLocation(State):
    def __init__(self, location):
        super(State, self).__init__()
        self.meta = BaseStateMeta()
        self.location = location

    def as_array(self):
        result = np.zeros((1, 2))
        result[self.location] = 1
        return result


class MockStateMachineSystem(StateMachineSystem):
    def __init__(self):
        super(MockStateMachineSystem, self).__init__()
        self.num_internal_states = 2
        self.num_actions = [3, 3]
        self.model = MockModel()
        self.policy = MockPolicy()
        self.action_value_function = MockActionValueFunction()
        self.reward_function = MockRewardFunction()


class MockPolicy(Policy):
    def calculate_action_value_probabilities(self, action_values):
        return np.array([1.0, 0.0])


class MockRewardFunction(RewardFunction):
    def __call__(self, old_state, action, new_state):
        if old_state.external_state.location == AT_HOME:
            return 0

        rewards = {
            (HUNGRY, AT_RESTAURANT, EAT): 10,
            (HUNGRY, AT_RESTAURANT, GO_TO_RESTAURANT): -1,
            (HUNGRY, AT_RESTAURANT, GO_TO_BAR): -1,
            (HUNGRY, AT_BAR, EAT): 5,
            (HUNGRY, AT_BAR, GO_TO_RESTAURANT): -1,
            (HUNGRY, AT_BAR, GO_TO_BAR): -1,
            (SATISFIED, AT_BAR, EAT): -2,
            (SATISFIED, AT_RESTAURANT, EAT): -1,
            (SATISFIED, AT_BAR, GO_HOME): 1,
        }

        k = (old_state.internal_state, old_state.external_state.location, action)

        return rewards.get(k, 0)


class MockActionValueFunction(ActionValueFunction):
    def __call__(self, state):
        # Arbitrary values, all different purely for testing
        action_values = {
            (HUNGRY, AT_RESTAURANT): np.array([1, 2, 3]),
            (HUNGRY, AT_BAR): np.array([4, 5, 6]),
            (SATISFIED, AT_RESTAURANT): np.array([7, 8, 9]),
            (SATISFIED, AT_BAR): np.array([10, 11, 12])
        }

        k = (state.internal_state, state.external_state.location)

        return action_values.get(k, 0)


class MockModel(Model):

    # Should we store the number of actions in here?
    # If we did that, we would need a different class for
    # the state machine models and normal models.
    def apply_action(self, state, action):
        if self.is_terminal(state):
            return state

        results = {
            (HUNGRY, AT_RESTAURANT, EAT): build_state(SATISFIED, AT_RESTAURANT),
            (HUNGRY, AT_RESTAURANT, GO_TO_BAR): build_state(HUNGRY, AT_BAR),
            (HUNGRY, AT_RESTAURANT, GO_TO_RESTAURANT): build_state(HUNGRY, AT_RESTAURANT),
            (HUNGRY, AT_BAR, EAT): build_state(SATISFIED, AT_BAR),
            (HUNGRY, AT_BAR, GO_TO_BAR): build_state(HUNGRY, AT_BAR),
            (HUNGRY, AT_BAR, GO_TO_RESTAURANT): build_state(HUNGRY, AT_RESTAURANT),
            (SATISFIED, AT_RESTAURANT, GO_HOME) : build_state(SATISFIED, AT_HOME),
            (SATISFIED, AT_RESTAURANT, GO_TO_RESTAURANT): build_state(SATISFIED, AT_RESTAURANT),
            (SATISFIED, AT_RESTAURANT, GO_TO_BAR): build_state(SATISFIED, AT_BAR),
            (SATISFIED, AT_BAR, GO_HOME) : build_state(SATISFIED, AT_HOME),
            (SATISFIED, AT_BAR, GO_TO_RESTAURANT) : build_state(SATISFIED, AT_RESTAURANT),
            (SATISFIED, AT_BAR, GO_TO_BAR) : build_state(SATISFIED, AT_BAR),
        }
        k = (state.internal_state, state.external_state.location, action)

        return results[k]

    def is_terminal(self, state):
        return state.external_state.location == AT_HOME


def build_state(internal_state, location):
    external_state = AgentLocation(location)
    return IntExtState(internal_state, external_state)
