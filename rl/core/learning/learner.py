from rl.core.learning.target_array_calculator import\
    SarsaActionTargetCalculator,\
    ExpectedSarsaActionTargetCalculator,\
    QLearningActionTargetCalculator

from rl.core.learning.target_array_calculator import\
    build_sarsa_target_array_calculator,\
    build_q_learning_target_array_calculator,\
    build_expected_sarsa_target_array_calculator,\
    build_vectorized_q_learning_target_array_calculator,\
    build_vectorized_expected_sarsa_target_array_calculator,\
    build_vectorized_sarsa_target_array_calculator


class Learner(object):

    def __init__(self, rl_system, target_array_calculator):
        self.rl_system = rl_system
        self.target_array_calculator = target_array_calculator

    def learn(self, experience, **kwargs):
        self.target_array_calculator.learn(experience, **kwargs)


class VectorizedLearner(Learner):

    def learn(self, experience, **kwargs):
        target_array = self.target_array_calculator.get_target_array(experience)
        self.rl_system.action_value_function.vectorized_fit(
            experience.get_training_states(),
            target_array,
            **kwargs)


class ScalarLearner(Learner):

    def learn(self, experience, **kwargs):
        target_array = self.target_array_calculator.get_target_array(experience)
        self.rl_system.action_value_function.scalar_fit(
            experience.get_training_states(),
            experience.get_training_actions(),
            target_array,
            **kwargs
        )


def build_sarsa_learner(rl_system, discount_factor=1.0):
    target_array_calculator = build_sarsa_target_array_calculator(rl_system, discount_factor)
    learner = ScalarLearner(rl_system, target_array_calculator=target_array_calculator)
    return learner


def build_q_learner(rl_system, discount_factor=1.0):
    target_array_calculator = build_q_learning_target_array_calculator(rl_system, discount_factor)
    learner = ScalarLearner(rl_system, target_array_calculator=target_array_calculator)
    return learner


def build_expected_sarsa_learner(rl_system, discount_factor=1.0):
    target_array_calculator = build_expected_sarsa_target_array_calculator(rl_system, discount_factor)
    learner = ScalarLearner(rl_system, target_array_calculator=target_array_calculator)
    return learner


def build_vectorized_sarsa_learner(rl_system, discount_factor=1.0):
    target_array_calculator = build_vectorized_sarsa_target_array_calculator(rl_system, discount_factor)
    learner = VectorizedLearner(rl_system, target_array_calculator=target_array_calculator)
    return learner


def build_vectorized_q_learner(rl_system, discount_factor=1.0):
    target_array_calculator = build_vectorized_q_learning_target_array_calculator(rl_system, discount_factor)
    learner = VectorizedLearner(rl_system, target_array_calculator=target_array_calculator)
    return learner


def build_vectorized_expected_sarsa_learner(rl_system, discount_factor=1.0):
    target_array_calculator = build_vectorized_expected_sarsa_target_array_calculator(rl_system, discount_factor)
    learner = VectorizedLearner(rl_system, target_array_calculator=target_array_calculator)
    return learner