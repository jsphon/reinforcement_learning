import unittest
import logging

from rl.grid_world import GridState
from rl.grid_world import GridRewardFunction, GridWorld, TabularValueFunction, GridState
from rl.policy import EquiProbableRandomPolicy
from rl.value import ValueFunction
from rl.experience import ExperienceGenerator, Episode
import numpy as np

N = 1000

logging.basicConfig(level=logging.DEBUG)


class GridWorldTests(unittest.TestCase):

    def test___init__(self):
        world = GridWorld()

    def test_evaluate_policy(self):
        world = GridWorld()
        world.policy = EquiProbableRandomPolicy(4)
        result = world.evaluate_policy()

        self.assertIsInstance(result, ValueFunction)

    def test_generate_episode(self):
        world = GridWorld()
        experience_generator = ExperienceGenerator(world)
        episode = experience_generator.generate_episode(100)

        logging.info(episode)
        logging.info('hello world')

        for state in episode.states:
            logging.info(state.player)

        for reward in episode.rewards:
            logging.info(reward)

    def test_get_state_array(self):
        states = [GridState((1, 1)), GridState((2, 2)), GridState((3, 3))]
        rewards = [-1, -1, 0]
        actions = [1, 2, 3]
        episode = Episode(states, actions, rewards)

        e0 = get_expected_vector((1, 1))
        e1 = get_expected_vector((2, 2))
        e2 = get_expected_vector((3, 3))
        expected = np.c_[e0, e1, e2].T

        logging.info(expected)

        state_array = episode.get_state_array()

        np.testing.assert_array_equal(expected, state_array)


def get_expected_vector(player):
    result = np.zeros(16, dtype=np.bool)
    result[4 * player[0] + player[1]] = True
    return result




if __name__ == '__main__':
    unittest.main()
