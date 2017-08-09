import logging

import numpy as np

from rl.core.experience import ExperienceGenerator, StatesList
from rl.core.learner import QLearner, SarsaLearner, ExpectedSarsaLearner
from rl.environments.cliff_world import CliffWorld, GridState
from rl.core.policy import SoftmaxPolicy
from rl.lib.timer import Timer

np.set_printoptions(precision=1)
np.set_printoptions(linewidth=200)
np.set_printoptions(suppress=True)

logging.basicConfig(level=logging.DEBUG)

grid_world = CliffWorld()
grid_world.policy.epsilon=0.5
generator = ExperienceGenerator(grid_world)
episode = generator.generate_episode()

learner = QLearner(grid_world)
#learner = SarsaLearner(grid_world)
#
# learner = ExpectedSarsaLearner(grid_world)
# learner.rl_system.policy = SoftmaxPolicy(grid_world)

with Timer('training') as t:

    for i in range(10):
        #grid_world.policy.epsilon *= 0.99
        #grid_world.policy.epsilon = max(grid_world.policy.epsilon, 0.05)
        #grid_world.policy.epsilon = 0.1
        print('=======ESPILON %s==========' % grid_world.policy.epsilon )

        #episode = generator.generate_episode()
        episodes = generator.generate_episodes(num_episodes=10, max_len=150)
        average_reward = np.mean([sum(episode.rewards) for episode in episodes])

        with Timer('Learning'):
            learner.learn(episodes, verbose=1, epochs=1)

        print('=== AVERAGE REWARD %s' % average_reward)
        print('=== Value Function %i ===' % i)
        print(grid_world.get_value_grid())

        print('=== Greedy Actions %i ===' % i)
        print(grid_world.get_greedy_action_grid_string())

#print('=== Reward Function ===')
#print(grid_world.get_reward_grid())

print('=== Value Function ===')
print(grid_world.get_value_grid())

print('=== Greedy Actions ===')
print(grid_world.get_greedy_action_grid_string())
