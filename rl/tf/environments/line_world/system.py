
from rl.tf.core.learning.learner import Learner
from rl.tf.core.policy import EpsilonGreedyPolicy
from rl.tf.core.system import System
from rl.tf.core.value_function import ValueFunctionBuilder
from rl.tf.environments.line_world.model import LineWorldModel
from rl.tf.core.learning.target_array_calculator import ModelBasedTargetArrayCalculator
from rl.tf.core.learning.action_target_calculator import QLearningActionTargetCalculator
from rl.tf.environments.line_world.reward_function import RewardFunction
from rl.lib.timer import Timer


class LineWorldSystem(System):

    def __init__(self):
        self.action_value_function = ValueFunctionBuilder(10, [10], 2, use_one_hot_input_transform=True)
        self.policy = EpsilonGreedyPolicy()
        self.model = LineWorldModel()
        self.reward_function = RewardFunction()


if __name__=='__main__':

    import numpy as np
    import tensorflow as tf

    a_states = np.arange(10)
    t_states = tf.constant(a_states, dtype=tf.int32)

    lws = LineWorldSystem()
    action_target_calculator = QLearningActionTargetCalculator(lws)

    calculator = ModelBasedTargetArrayCalculator(lws, action_target_calculator)

    t_action_values = lws.action_value_function.vectorized(t_states)

    learner = Learner(calculator, t_states)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    with Timer('Training...'):
        learner.train(sess, num_epochs=100)
