import unittest
import logging

from rl.grid_world import GridState
from rl.learner import Learner, RewardLearner, QLearner
from rl.grid_world import GridRewardFunction, GridWorld, GridState
from rl.policy import EquiProbableRandomPolicy
from rl.value import ValueFunction
from rl.experience import ExperienceGenerator, Episode
import numpy as np

N = 1000

logging.basicConfig(level=logging.DEBUG)


class RewardLearnerTests(np.testing.TestCase):

    def test_learn_episode(self):
        states = [GridState((1, 1)), GridState((2, 2)), GridState((3, 3))]
        rewards = [-1, -1, 0]
        actions = [1, 2, 3]
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


class QLearnerTests(unittest.TestCase):

    def test_learn_episode(self):
        states = [GridState((1, 1)), GridState((2, 2)), GridState((3, 3))]
        rewards = [-1, -1, 0]
        actions = [1, 2, 3]
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


def get_expected_vector(player):
    result = np.zeros(16, dtype=np.bool)
    result[4 * player[0] + player[1]] = True
    return result




if __name__ == '__main__':
    unittest.main()
