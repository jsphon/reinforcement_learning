import logging
import unittest

import numpy as np

from rl.core import RLSystem, State
from rl.core.experience import Episode
from rl.core.learner import RewardLearner, QLearner, VectorQLearner, SarsaLearner, ExpectedSarsaLearner, \
    VectorSarsaLearner
from rl.core.policy import Policy
from rl.core.reward_function import RewardFunction
from rl.core.value_function import ActionValueFunction
from rl.environments.grid_world import GridWorld, GridState

N = 1000

logging.basicConfig(level=logging.DEBUG)


class RewardLearnerTests(np.testing.TestCase):

    def test_learn(self):
        np.random.seed(1)
        states = [GridState((1, 1)), GridState((2, 2)), GridState((3, 3))]
        rewards = [-1, -1]
        actions = [1, 2]
        episode = Episode(states, actions, rewards)

        grid_world = GridWorld()

        y = grid_world.action_value_function.on_list(episode.states)
        logging.info('Reward Learner initial targets are:')
        logging.info('\n' + str(y))

        learner = RewardLearner(grid_world)
        learner.learn(episode)

        y = grid_world.action_value_function.on_list(episode.states)
        logging.info('Reward Learner fitted targets are:')
        logging.info('\n' + str(y))

        np.testing.assert_allclose([0, -1, 0, 0], y[0], atol=0.5)
        np.testing.assert_allclose([0, 0, -1, 0], y[1], atol=0.5)
        np.testing.assert_allclose([0, 0, 0, 0], y[2], atol=0.5)

    def test_get_target_array(self):
        mock_system = MockSystem()
        learner = RewardLearner(mock_system)

        states = [MockState1(), MockState2A(), MockState3()]
        actions = [0, 0]
        rewards = [0, 0]
        episode = Episode(states=states, actions=actions, rewards=rewards)

        targets = learner.get_target_array(episode)
        logging.info('get targets = %s' % str(targets))

        expected = np.array([[0.0, 0.0], [0.0, 0.0]])
        np.testing.assert_almost_equal(expected, targets)


class QLearnerTests(unittest.TestCase):

    def test_learn(self):
        np.random.seed(1)
        states = [GridState((1, 1)), GridState((2, 2)), GridState((3, 3))]
        rewards = [-1, -1]
        actions = [1, 2]
        episode = Episode(states, actions, rewards)

        grid_world = GridWorld()

        y = grid_world.action_value_function.on_list(episode.states)
        logging.info('Q Learning initial targets are:')
        logging.info('\n' + str(y))

        learner = QLearner(grid_world)
        learner.learn(episode)

        y = grid_world.action_value_function.on_list(episode.states)
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

        targets = learner.get_target_array(episode)

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

        targets = learner.get_target_array(episode)

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


class TestVectorSarsaLearner(unittest.TestCase):
    def test_get_targets(self):
        mock_system = MockSystem()
        learner = VectorSarsaLearner(mock_system)

        states = [MockState1(), MockState2A()]
        actions = [0]
        rewards = [0]
        episode = Episode(states=states, actions=actions, rewards=rewards)

        targets = learner.get_targets(episode)

        # target[0] should be
        # reward + Q(next state=MockState2A)[0] = 11 + [1, 0][0] = 12
        # target[1] should be
        # reward(action1) + Q(next state)[0] = 22 + [0, 1][0] = 22
        # target[1] should be Reward of taking action1 from MockState1 plus Q(MockState2B)
        # Reward of 33 for the new state, then action reward is always zero
        # So target[1] = [0, 0]

        expected = np.array([12, 22])
        np.testing.assert_almost_equal(expected, targets[0])

        expected = np.array([33, 33])
        np.testing.assert_almost_equal(expected, targets[1])


class TestVectorQLearner(unittest.TestCase):
    def test_get_state_targets1(self):
        mock_system = MockSystem()
        learner = VectorQLearner(mock_system)

        targets = learner.get_state_targets(MockState1())

        expected = np.array([12, 23])
        np.testing.assert_almost_equal(expected, targets)

    def test_get_state_targets2A(self):
        mock_system = MockSystem()
        learner = VectorQLearner(mock_system)

        # Expectations
        # Action 0
        #  33 + 1.0 * 0 = 33
        # Action 1
        # 33 + 1.0 * 0 = 33
        expected = np.array([33, 33])
        targets = learner.get_state_targets(MockState2A())

        np.testing.assert_almost_equal(expected, targets)

    def test_get_targets(self):
        mock_system = MockSystem()
        learner = VectorQLearner(mock_system)

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


class SarsaTests(unittest.TestCase):
    def test_get_state_targets1(self):
        mock_system = MockSystem()
        learner = SarsaLearner(mock_system, gamma=0.9)

        targets = learner.get_state_targets(state=MockState1(),
                                            action=0,
                                            reward=11)

        # Expected0 = reward(action|state) + gamma * Q(next_state|next_action)
        #           = 11 + 0.9 * 1
        #           = 11.9
        # Expected1 = Q(MockState1(), action=1)
        #           = 2
        expected = np.array([11.9, 2.0])
        np.testing.assert_almost_equal(expected, targets)

    def test_get_state_targets2A(self):
        mock_system = MockSystem()
        learner = SarsaLearner(mock_system, gamma=0.9)

        targets = learner.get_state_targets(state=MockState2A(),
                                            action=0,
                                            reward=1)

        # Expected0 = reward(action|state) + gamma * Q(next_state|next_action)
        #           = 1 + 0.9 * 0
        #           = 1.0
        # Expected1 = Q(MockState2A(), action=1)
        #           = 0.0
        expected = np.array([1.0, 0.0])
        np.testing.assert_almost_equal(expected, targets)

    def test_get_state_targets2B(self):
        mock_system = MockSystem()
        learner = SarsaLearner(mock_system, gamma=0.9)

        targets = learner.get_state_targets(state=MockState2B(),
                                            action=0,
                                            reward=2)

        # Expected0 = reward(action|state) + gamma * Q(next_state|next_action)
        #           = 2 + 0.9 * 0
        #           = 2.0
        # Expected1 = Q(MockState2B(), action=1)
        #           = 1.0
        expected = np.array([2.0, 1.0])
        np.testing.assert_almost_equal(expected, targets)

    def test_get_targets1(self):
        states = [MockState1(), MockState2A(), MockState3()]
        rewards = [11, 33]
        actions = [0, 0]
        episode = Episode(states, actions, rewards)
        learner = SarsaLearner(MockSystem(), gamma=0.9)

        targets = learner.get_target_array(episode)
        # [11.0 * 0.9 * 1.0, 2.0]
        expected0 = np.array([11.9, 2.0])
        np.testing.assert_almost_equal(expected0, targets[0])

        # [33.0 * 0.9 * 0.0, 0.0]
        expected0 = np.array([33.0, 0.0])
        np.testing.assert_almost_equal(expected0, targets[1])

    def test_get_targets2(self):
        states = [MockState1(), MockState2B(), MockState3()]
        rewards = [22, 33]
        actions = [1, 0]
        episode = Episode(states, actions, rewards)
        learner = SarsaLearner(MockSystem(), gamma=0.9)

        targets = learner.get_target_array(episode)
        # [2.0, 22.0 + 0.9 * 0]
        expected0 = np.array([1.0, 22.0])
        np.testing.assert_almost_equal(expected0, targets[0])

        # [33 + 0.9 * 0.0, 1.0]
        expected0 = np.array([33.0, 1.0])
        np.testing.assert_almost_equal(expected0, targets[1])


class MockSystem(RLSystem):
    def __init__(self):
        super(MockSystem, self).__init__()
        self.num_actions = 2
        self.model = MockModel()
        self.policy = MockPolicy(self)
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


if __name__ == '__main__':
    unittest.main()
