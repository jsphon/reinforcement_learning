from rl.core.learning.target_array_calculator import build_target_array_calculator


class Learner(object):

    def __init__(self, rl_system, target_array_calculator):
        self.rl_system = rl_system
        self.target_array_calculator = target_array_calculator


class VectorizedLearner(Learner):

    def learn(self, states, **kwargs):
        target_array = self.target_array_calculator.get_target_matrix(states)
        self.rl_system.action_value_function.vectorized_fit(
            states,
            target_array,
            **kwargs)


# class ScalarLearner(Learner):
#
#     def learn(self, experience, **kwargs):
#         target_array = self.target_array_calculator.get_target_matrix(experience)
#         self.rl_system.action_value_function.scalar_fit(
#             experience.get_training_states(),
#             experience.get_training_actions(),
#             target_array,
#             **kwargs
#         )


def build_learner(rl_system, discount_factor, learning_algo='qlearning', calculator_type='modelbased'):
    target_array_calculator = build_target_array_calculator(rl_system,
                                                            discount_factor=discount_factor,
                                                            learning_algo=learning_algo,
                                                            calculator_type=calculator_type)
    learner = VectorizedLearner(rl_system, target_array_calculator)
    return learner
