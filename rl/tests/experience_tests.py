import unittest
import logging

from rl.grid_world import GridState
from rl.grid_world import GridRewardFunction, GridWorld, TabularValueFunction
from rl.policy import EquiProbableRandomPolicy
from rl.value import ValueFunction
from rl.experience import ExperienceGenerator

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



if __name__ == '__main__':
    unittest.main()
