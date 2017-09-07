import numpy as np

from rl.core import RLSystem, State
from rl.core.policy import Policy
from rl.core.reward_function import RewardFunction
from rl.core.value_function import ActionValueFunction


class MockSystem(RLSystem):
    def __init__(self):
        super(MockSystem, self).__init__()
        self.num_actions = 2
        self.model = MockModel()
        self.policy = MockPolicy()
        self.action_value_function = MockActionValueFunction(self.policy)
        self.reward_function = MockRewardFunction()


class MockPolicy(Policy):
    def calculate_action_value_probabilities(self, action_values):
        return np.array([1.0, 0.0])


class MockRewardFunction(RewardFunction):
    def __call__(self, old_state, action, new_state):

        if isinstance(old_state, MockState3) and isinstance(new_state, MockState3):
            # Terminal State
            return 0
        elif isinstance(new_state, MockState2A):
            return 11
        elif isinstance(new_state, MockState2B):
            return 22
        elif isinstance(new_state, MockState3):
            return 33


class MockActionValueFunction(ActionValueFunction):
    def __call__(self, state):

        if isinstance(state, MockState1):
            return np.array([1., 2.])
        elif isinstance(state, MockState2A):
            return np.array([1., 0.])
        elif isinstance(state, MockState2B):
            return np.array([0., 1.])
        elif isinstance(state, MockState3):
            return np.array([0., 0.])
        else:
            raise ValueError('State not expected %s' % str(state))


class MockModel(object):
    def apply_action(self, state, action):

        if isinstance(state, MockState1):
            if action == 0:
                return MockState2A()
            elif action == 1:
                return MockState2B()
            else:
                raise ValueError('Expect 0 or 1')
        elif isinstance(state, (MockState2A, MockState2B)):
            return MockState3()
        elif isinstance(state, MockState3):
            return MockState3()
        else:
            raise ValueError('object %s not expected' % str(state))


class MockState(State):
    def __init__(self):
        super(MockState, self).__init__()
        self.size = 3
        self.array_dtype = np.bool


class MockState1(MockState):
    def as_array(self):
        return np.array([False, False], np.bool).reshape(1, 2)


class MockState2A(MockState):
    def as_array(self):
        return np.array([True, False], np.bool).reshape(1, 2)


class MockState2B(MockState):
    def as_array(self):
        return np.array([False, True], np.bool).reshape(1, 2)


class MockState3(MockState):
    def as_array(self):
        return np.array([True, True], np.bool).reshape(1, 2)
