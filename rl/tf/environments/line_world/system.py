

from rl.tf.core.policy import EpsilonGreedyPolicy
from rl.tf.core.system import System
from rl.tf.core.value_function import ValueFunctionBuilder
from rl.tf.environments.line_world.model import LineWorldModel
from rl.tf.core.learning.target_array_calculator import ModelBasedTargetArrayCalculator
from rl.tf.core.learning.action_target_calculator import QLearningActionTargetCalculator
from rl.tf.environments.line_world.reward_function import RewardFunction

class LineWorldSystem(System):

    def __init__(self):
        self.action_value_function = ValueFunctionBuilder(10, [8, 6], 2, use_one_hot_input_transform=True)
        self.policy = EpsilonGreedyPolicy()
        self.model = LineWorldModel()
        self.reward_function = RewardFunction()


if __name__=='__main__':

    import numpy as np
    import tensorflow as tf
    from rl.tests.tf.utils import evaluate_tensor

    a_states = np.arange(10)
    #t_states = tf.Variable(a_states, dtype=tf.int32)
    #t_states = tf.constant(a_states, shape=(10, 1), dtype=tf.int32)
    t_states = tf.constant(a_states, dtype=tf.int32)

    lws = LineWorldSystem()
    action_target_calculator = QLearningActionTargetCalculator(lws)

    calculator = ModelBasedTargetArrayCalculator(lws, action_target_calculator)

    action_values = lws.action_value_function.calculate(t_states)
    print(evaluate_tensor(action_values))
    t_targets = calculator.get_states_targets(t_states)
    a_targets = evaluate_tensor(t_targets)
    print(a_targets)

    loss = tf.reduce_mean(tf.abs(action_values-t_targets))

    train_op = lws.action_value_function.train_op(t_states, t_targets, learning_rate=0.001)
    #train_loop = lws.action_value_function.train_loop(t_states, t_targets)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            sess.run([train_op])
            new_loss = sess.run(loss)
            print('new loss is %s' % new_loss)

            if (i%10)==0:

                print('After training, our action values are:')
                print(sess.run(action_values))

                print('targets:')
                print(sess.run(t_targets))


    """
    
    
    next_states = lws.model.apply_actions(t_states[0])
    rewards = lws.reward_function.state_rewards(t_states[0], next_states)
    next_states_action_values = lws.action_value_function.vectorized(tf.reshape(next_states, (2, 1)))

    print('Next states are of shape %s' % str(next_states.shape))
    print(evaluate_tensor(next_states))
    
    """

