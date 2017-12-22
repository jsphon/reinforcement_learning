
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
        self.action_value_function = ValueFunctionBuilder(10, [50, 20], 2, use_one_hot_input_transform=True)
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

    learner = Learner(calculator, t_states, vectorize=False)
    learner_v = Learner(calculator, t_states,
                        vectorize=True,
                        train_loop_steps=100,
                        learning_rate=0.01)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        print('=========== %s ============' % (i+1))
        with Timer('Train Loop...'):
            learner_v.run_train_loop(sess)
        print(learner.loss.eval())
        print('Action vaues')
        print(t_action_values.eval())
        print('Targets')
        print(calculator.get_states_targets_vectorized(t_states).eval())

