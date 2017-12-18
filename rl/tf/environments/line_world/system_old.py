

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
    from rl.tests.tf.utils import evaluate_tensor

    a_states = np.arange(10)
    t_states = tf.constant(a_states, dtype=tf.int32)

    lws = LineWorldSystem()
    action_target_calculator = QLearningActionTargetCalculator(lws)

    calculator = ModelBasedTargetArrayCalculator(lws, action_target_calculator)

    t_action_values = lws.action_value_function.vectorized(t_states)
    t_action_values_0_0 = lws.action_value_function.calculate(t_states[0])

    t_targets = calculator.get_states_targets(t_states)
    t_target0 = calculator.get_state_action_target(t_states[0], 0)
    t_target1 = calculator.get_state_action_target(t_states[0], 1)

    t_next_state = lws.model.apply_action(t_states[0], 0)
    t_next_state_action_values = lws.action_value_function.calculate(t_next_state)
    t_reward = lws.reward_function.action_reward(t_states[0], 0, t_next_state)

    #loss = tf.reduce_mean(tf.abs(t_action_values-t_targets))
    loss = lws.action_value_function.squared_loss(t_states, t_targets)

    t_learning_rate = tf.Variable(0.001, dtype=tf.float32, name='learning_rate')
    assign_op = t_learning_rate.assign(0.999 * t_learning_rate)

    #train_op = lws.action_value_function.train_op(t_states, t_targets, learning_rate=0.01)
    train_op = lws.action_value_function.train_op(t_states[3:], t_targets[3:], learning_rate=0.01)


    #train_loop = lws.action_value_function.train_loop(t_states, t_targets, num_steps=100, learning_rate=0.001)
    g = calculator.get_state_action_target_graph(t_states[0], 0)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    initial_action_values = t_action_values.eval()
    initial_targets = t_targets.eval()

    print('Initial action values are\n%s' % str(initial_action_values))
    print('Initial Targets are\n%s' % str(initial_targets))

    diff = np.abs((initial_action_values[0].max() - 1) - initial_targets[0, 0])
    assert diff < 1e-6

    sess.run([train_op])

    trained_action_values = t_action_values.eval()
    trained_targets = t_targets.eval()
    trained_targets0 = t_target0.eval()

    diff = np.abs((trained_action_values[0].max() - 1) - trained_targets[0, 0])
    #assert diff < 1e-6

    d = (-1 + g.next_state_action_values.eval().max()) - g.non_terminal_targets.eval()

    manual_q = g.reward.eval() + g.next_state_action_values.eval().max()
    print( 'manual q calc %s' % manual_q )

    actual_q = action_target_calculator.calculate(g.reward, g.next_state_action_values).eval()
    print( 'actual q calc %s' % actual_q)


    def train_n_times(n):
        with Timer('Looping %i times' % n):
            for i in range(n):

                #ix = np.random.choice(10, 5)
                #train_op = lws.action_value_function.train_op(t_states[:3], t_targets[:3], learning_rate=0.01)

                sess.run([train_op])

                if (i % 100==0) or i==n-1:
                    print(i, 'new loss is %s' % loss.eval())
                    print('learning rate is %0.4f' % t_learning_rate.eval())

                    print('After training, our action values are:')
                    print(t_action_values.eval())

                    print('targets:')
                    print(t_targets.eval())

    train_n_times(1000)
