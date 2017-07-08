import logging
import unittest

import numpy as np

from rl.core import RLSystem, State
from rl.experience import Episode
from rl.grid_world import GridWorld, GridState
from rl.learner import RewardLearner, QLearner
from rl.policy import Policy
from rl.value import ActionValueFunction

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

        states = [MockState1(), MockState1A(), MockState2()]
        actions = [0, 0]
        rewards = [0, 0]
        episode = Episode(states=states, actions=actions, rewards=rewards)

        targets = learner.get_targets(episode)
        logging.info('get targets = %s' % str(targets))

        expected = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
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

        states = [MockState1(), MockState1A(), MockState2()]
        actions = [0, 0]
        rewards = [0, 0]
        episode = Episode(states=states, actions=actions, rewards=rewards)

        targets = learner.get_targets(episode)

        expected = np.array([1, 2, 3])
        np.testing.assert_almost_equal(expected, targets[0])

        expected = np.array([0, 0, 0])
        np.testing.assert_almost_equal(expected, targets[1])

    def test_get_targets2(self):
        mock_system = MockSystem()
        learner = QLearner(mock_system)

        states = [MockState1(), MockState1B(), MockState2()]
        actions = [1, 0]
        rewards = [0, 10]
        episode = Episode(states=states, actions=actions, rewards=rewards)

        targets = learner.get_targets(episode)

        # Elements 0 and 2 come from MockState1's action value (1 and 3)
        # Element 1 is the reward at 1 (0) plus the best action value
        # from MockState1B (1). So it should be 1
        expected = np.array([1, 1, 3])
        np.testing.assert_almost_equal(expected, targets[0])

        # Elements 1 and 2 come from MockState1B's action value (1 and 0)
        # Element 0 comes from the sum of the 2nd reward (10) and MockState2's
        # best action value (0)
        expected = np.array([10, 1, 0])
        np.testing.assert_almost_equal(expected, targets[1])


class MockSystem(RLSystem):
    def __init__(self):
        super(MockSystem, self).__init__()
        self.num_actions = 3
        self.model = MockModel()
        self.policy = MockPolicy(self.num_actions)
        self.action_value_function = MockActionValueFunction(self.policy)


class MockPolicy(Policy):

    def __call__(self, state):
        return np.array([1.0, 0.0, 0.0])


class MockActionValueFunction(ActionValueFunction):

    def __call__(self, state_vector):

        if np.array_equal(state_vector, MockState1().as_array()):
            return np.array([1, 2, 3])
        elif np.array_equal(state_vector, MockState1A().as_array()):
            return np.array([1, 0, 0])
        elif np.array_equal(state_vector, MockState1B().as_array()):
            return np.array([0, 1, 0])
        elif np.array_equal(state_vector, MockState1C().as_array()):
            return np.array([0, 0, 1])
        elif np.array_equal(state_vector, MockState2().as_array()):
            return np.array([0, 0, 0])
        else:
            raise ValueError('State not expected %s' % str(state_vector))


class MockModel(object):

    def apply_action(self, state, action):

        if isinstance(state, MockState1):
            if action == 0:
                return MockState1A()
            elif action == 1:
                return MockState1B()
            elif action == 2:
                return MockState1C()
            else:
                raise ValueError('Expect 0, 1 or 2')
        elif isinstance(state, (MockState1A, MockState1B, MockState1C)):
            return MockState2()
        elif isinstance(state, MockState2):
            return MockState2()
        else:
            raise ValueError('object %s not expected' % str(state))


class MockState(State):
    def __init__(self):
        super(MockState, self).__init__()
        self.size = 3
        self.array_dtype = np.bool


class MockState1(MockState):
    def as_array(self):
        return np.array([False, False, False], np.bool).reshape(1, 3)


class MockState1A(MockState):
    def as_array(self):
        return np.array([True, False, False], np.bool).reshape(1, 3)


class MockState1B(MockState):
    def as_array(self):
        return np.array([False, True, False], np.bool).reshape(1, 3)


class MockState1C(MockState):
    def as_array(self):
        return np.array([False, False, True], np.bool).reshape(1, 3)


class MockState2(MockState):
    def as_array(self):
        return np.array([True, True, True], np.bool).reshape(1, 3)


def get_expected_vector(player):
    result = np.zeros(16, dtype=np.bool)
    result[4 * player[0] + player[1]] = True
    return result


if __name__ == '__main__':
    unittest.main()
