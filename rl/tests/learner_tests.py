import logging
import unittest

import numpy as np

from rl.core import RLSystem, State
from rl.experience import Episode
from rl.grid_world import GridWorld, GridState
from rl.learner import RewardLearner, QLearner, WQLearner
from rl.policy import Policy
from rl.value import ActionValueFunction
from rl.reward_function import RewardFunction

N = 1000

logging.basicConfig(level=logging.DEBUG)


class RewardLearnerTests(np.testing.TestCase):

    def test_learn_episode(self):

        np.random.seed(1)
        states = [GridState((1, 1)), GridState((2, 2)), GridState((3, 3))]
        rewards = [-1, -1]
        actions = [1, 2]
        episode = Episode(states, actions, rewards)

        grid_world = GridWorld()

        y = grid_world.action_value_function(episode.get_state_array())
        logging.info('Reward Learner initial targets are:')
        logging.info('\n' + str(y))

        learner = RewardLearner(grid_world)
        learner.learn_episode(episode)

        y = grid_world.action_value_function(episode.get_state_array())
        logging.info('Reward Learner fitted targets are:')
        logging.info('\n' + str(y))

        np.testing.assert_allclose([0, -1, 0, 0], y[0], atol=0.5)
        np.testing.assert_allclose([0, 0, -1, 0], y[1], atol=0.5)
        np.testing.assert_allclose([0, 0, 0, 0], y[2], atol=0.5)

    def test_get_targets(self):

        mock_system = MockSystem()
        learner = RewardLearner(mock_system)

        states = [MockState1(), MockState2A(), MockState3()]
        actions = [0, 0]
        rewards = [0, 0]
        episode = Episode(states=states, actions=actions, rewards=rewards)

        targets = learner.get_targets(episode)
        logging.info('get targets = %s' % str(targets))

        expected = np.array([[0.0, 0.0], [0.0, 0.0]])
        np.testing.assert_almost_equal(expected, targets)


class QLearnerTests(unittest.TestCase):

    def test_learn_episode(self):
        np.random.seed(1)
        states = [GridState((1, 1)), GridState((2, 2)), GridState((3, 3))]
        rewards = [-1, -1]
        actions = [1, 2]
        episode = Episode(states, actions, rewards)

        grid_world = GridWorld()

        y = grid_world.action_value_function(episode.get_state_array())
        logging.info('Q Learning initial targets are:')
        logging.info('\n' + str(y))

        learner = QLearner(grid_world)
        learner.learn_episode(episode)

        y = grid_world.action_value_function(episode.get_state_array())
        logging.info('Q Learning fitted targets are:')
        logging.info('\n' + str(y))

        np.testing.assert_allclose([0, -1, 0, 0], y[0], atol=0.5)
        np.testing.assert_allclose([0, 0, -1, 0], y[1], atol=0.5)
        np.testing.assert_allclose([0, 0, 0, 0], y[2], atol=0.5)

    def test_get_targets(self):
        mock_system = MockSystem()
        learner = QLearner(mock_system)

        states = [MockState1(), MockState2A(), MockState3()]
        actions = [0, 0]
        rewards = [0, 0]
        episode = Episode(states=states, actions=actions, rewards=rewards)

        targets = learner.get_targets(episode)

        expected = np.array([1, 2])
        np.testing.assert_almost_equal(expected, targets[0])

        expected = np.array([0, 0])
        np.testing.assert_almost_equal(expected, targets[1])

    def test_get_targets2(self):
        mock_system = MockSystem()
        learner = QLearner(mock_system)

        states = [MockState1(), MockState2B(), MockState3()]
        actions = [1, 0]
        rewards = [0, 10]
        episode = Episode(states=states, actions=actions, rewards=rewards)

        targets = learner.get_targets(episode)

        # Elements 0 and 2 come from MockState1's action value (1 and 3)
        # Element 1 is the reward at 1 (0) plus the best action value
        # from MockState1B (1). So it should be 1
        expected = np.array([1, 1])
        np.testing.assert_almost_equal(expected, targets[0])

        # Elements 1 and 2 come from MockState1B's action value (1 and 0)
        # Element 0 comes from the sum of the 2nd reward (10) and MockState2's
        # best action value (0)
        expected = np.array([10, 1])
        np.testing.assert_almost_equal(expected, targets[1])


class TestWQLearner(unittest.TestCase):

    def test_get_state_targets1(self):

        mock_system = MockSystem()
        learner = WQLearner(mock_system)

        targets = learner.get_state_targets(MockState1())

        expected = np.array([12, 23])
        np.testing.assert_almost_equal(expected, targets)

    def test_get_state_targets2A(self):

        mock_system = MockSystem()
        learner = WQLearner(mock_system)

        #Expectations
        # Action 0
        #  33 + 1.0 * 0 = 33
        # Action 1
        # 33 + 1.0 * 0 = 33
        expected = np.array([33, 33])
        targets = learner.get_state_targets(MockState2A())

        np.testing.assert_almost_equal(expected, targets)

    def test_get_targets(self):
        mock_system = MockSystem()
        learner = WQLearner(mock_system)

        states = [MockState1(), MockState2A()]
        actions = [0]
        rewards = [0]
        episode = Episode(states=states, actions=actions, rewards=rewards)

        targets = learner.get_targets(episode)

        # target[0] should be
        # reward + max(Q(next state=MockState2A)) = 11 + max(1, 0) = 12
        # target[1] should be
        # reward(action1) + max(Q(next state)) = 22 + max(0, 1) = 1
        # target[1] should be Reward of taking action1 from MockState1 plus Q(MockState2B)

        expected = np.array([12, 23])
        np.testing.assert_almost_equal(expected, targets[0])

        expected = np.array([33, 33])
        np.testing.assert_almost_equal(expected, targets[1])


class MockSystem(RLSystem):
    def __init__(self):
        super(MockSystem, self).__init__()
        self.num_actions = 2
        self.model = MockModel()
        self.policy = MockPolicy(self.num_actions)
        self.action_value_function = MockActionValueFunction(self.policy)
        self.reward_function = MockRewardFunction()


class MockPolicy(Policy):

    def __call__(self, state):
        return np.array([1.0, 0.0])


class MockRewardFunction(RewardFunction):

    def __call__(self, old_state, action, new_state):

        if isinstance(new_state, MockState2A):
            return 11
        elif isinstance(new_state, MockState2B):
            return 22
        elif isinstance(new_state, MockState3):
            return 33


class MockActionValueFunction(ActionValueFunction):

    def __call__(self, state_vector):

        if np.array_equal(state_vector, MockState1().as_array()):
            return np.array([1, 2])
        elif np.array_equal(state_vector, MockState2A().as_array()):
            return np.array([1, 0])
        elif np.array_equal(state_vector, MockState2B().as_array()):
            return np.array([0, 1])
        elif np.array_equal(state_vector, MockState3().as_array()):
            return np.array([0, 0])
        else:
            raise ValueError('State not expected %s' % str(state_vector))


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


if __name__ == '__main__':
    unittest.main()
