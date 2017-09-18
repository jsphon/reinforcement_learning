import logging
import unittest

import numpy as np

from rl.core.experience import Episode
from rl.environments.simple_grid_world import SimpleGridWorld, SimpleGridState
from rl.core.learning.learner import build_q_learner, build_vectorized_q_learner

N = 1000

logging.basicConfig(level=logging.DEBUG)


class QLearningTests(unittest.TestCase):

    def test_learn(self):

        np.random.seed(1)
        states = [SimpleGridState((1, 1)), SimpleGridState((2, 2)), SimpleGridState((3, 3))]
        rewards = [-1, -2]
        actions = [1, 2]
        episode = Episode(states, actions, rewards)

        grid_world = SimpleGridWorld()

        q_learner = build_q_learner(grid_world)
        q_learner.learn(episode)

        y = grid_world.action_value_function.on_list(episode.states)
        logging.info('Q Learning fitted targets are:')
        logging.info('\n' + str(y))

        # The learning rate is set to 0.1 by default, so we can calculate
        # the expect result usign the actions and rewards above.
        expected = np.array([
            [0, -0.1, 0, 0],
            [0, 0, -0.2, 0],
            [0, 0, 0, 0 ]
        ])

        np.testing.assert_array_equal(expected, y)


class VectorizedQLearningTests(unittest.TestCase):
    # This uses the model to generate the next step actions and rewards
    # so the learn function will only take a list of states.

    def test_learn(self):
        np.random.seed(1)
        states = [SimpleGridState((1, 1)), SimpleGridState((2, 2)), SimpleGridState((2, 3))]

        grid_world = SimpleGridWorld()

        q_learner = build_vectorized_q_learner(grid_world)
        q_learner.learn(states)

        y = grid_world.action_value_function.on_list(states)
        logging.info('Q Learning fitted targets are:')
        logging.info('\n' + str(y))

        # The learning rate is set to 0.1 by default, so we can calculate
        # the expect result usign the actions and rewards above.
        expected = np.array([
            [-0.1, -0.1, -0.1, -0.1], # Loses 1 in all directions
            [-0.1, -0.1, -0.1, -0.1], # Loses 1 in all directions
            [-0.1,  0.0, -0.1, -0.1] # Loses 1 in all but the down direction
        ])

        np.testing.assert_array_equal(expected, y)


if __name__ == '__main__':
    unittest.main()
