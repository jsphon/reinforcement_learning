import logging
import unittest

import numpy as np

from rl.core.experience import ExperienceGenerator, Episode
from rl.environments.grid_world import GridWorld, GridState

N = 1000

logging.basicConfig(level=logging.DEBUG)


class ExperienceGeneratorTests(unittest.TestCase):

    def test_generate_episode(self):
        world = GridWorld()
        experience_generator = ExperienceGenerator(world)
        episode = experience_generator.generate_episode(100)

        self.assertIsInstance(episode, Episode)

    def test_generate_episodes(self):

        world = GridWorld()
        experience_generator = ExperienceGenerator(world)
        episodes = experience_generator.generate_episodes(10)

        self.assertEqual(10, len(episodes))
        for episode in episodes:
            self.assertIsInstance(episode, Episode)


class EpisodeTests(unittest.TestCase):

    def test_get_state_array(self):
        states = [GridState((1, 1)), GridState((2, 2)), GridState((3, 3))]
        rewards = [-1, -1]
        actions = [1, 2]
        episode = Episode(states, actions, rewards)

        e0 = get_expected_vector((1, 1))
        e1 = get_expected_vector((2, 2))
        expected = np.c_[e0, e1].T

        logging.info(expected)

        state_array = episode.get_training_states_array()

        np.testing.assert_array_equal(expected, state_array)


def get_expected_vector(player):
    result = np.zeros(16, dtype=np.bool)
    result[4 * player[0] + player[1]] = True
    return result


if __name__ == '__main__':
    unittest.main()
