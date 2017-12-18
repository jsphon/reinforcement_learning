

class Learner(object):

    def __init__(self, calculator, t_states, learning_rate=0.01):
        self.calculator = calculator
        self.t_targets = self.calculator.get_states_targets(t_states)
        self.train_op = calculator.rl_system.action_value_function.train_op(t_states, self.t_targets, learning_rate=learning_rate)
        self.train_loop = calculator.rl_system.action_value_function.train_loop(t_states, self.t_targets, learning_rate=learning_rate)
        self.loss = self.calculator.rl_system.action_value_function.squared_loss(t_states, self.t_targets)

    def train(self, sess, num_epochs=1):
        for _ in range(num_epochs):
            sess.run(self.train_op)