import numpy as np


# Policies can be deterministic or stochastic.
# For deterministic policies, there is only one state taken, so we only need \pi(s) to tell
# us what happens
# For stochastic policies, there is a probability of taking each action a,
# so we have \pi(a|s), though perhaps we should just return a vector of probabilities
# per action, to give us the benefits of vectorised calculations


class Policy(object):
    """


    "A policy, \pi, is a mapping from each state, s\in S, and action a\in A(s), to the probability
    \pi(a|s) of taking action a when in state s", Sutton 3.7 Value Functions

    """

    def __init__(self, rl_system):
        self.rl_system = rl_system

    def calculate_state_probabilities(self, state):
        action_values = self.rl_system.action_value_function(state)
        return self.calculate_action_value_probabilities(action_values)

    def calculate_action_value_probabilities(self, action_values):
        raise NotImplemented()

    def choose_action(self, state):
        probabilities = self.calculate_state_probabilities(state)
        num_actions = self.rl_system.num_actions
        action = np.random.choice(num_actions, p=probabilities)
        return action


class DeterministicPolicy(Policy):
    """
    Known as \pi(s) in the literature

    Is this even needed? We could just use a stochastic policy that returns a 1.0 where
    the probability of transition is 100%
    """

    pass


class StochasticPolicy(Policy):
    """
    Known as \pi(s) in the literature
    """

    pass


class EquiProbableRandomPolicy(StochasticPolicy):
    def calculate_action_value_probabilities(self, action_values):
        num_actions = self.rl_system.num_actions
        return np.ones(num_actions) / num_actions


class EpsilonGreedyPolicy(StochasticPolicy):
    def __init__(self, rl_system, epsilon=0.1):
        super(EpsilonGreedyPolicy, self).__init__(rl_system)
        self.epsilon = epsilon

    def calculate_action_value_probabilities(self, action_values):
        best_action = action_values.argmax()

        num_actions = self.rl_system.num_actions
        probabilities = np.ones(num_actions) * (self.epsilon) / (num_actions - 1)
        probabilities[best_action] = 1.0 - self.epsilon

        return probabilities


class SoftmaxPolicy(StochasticPolicy):
    def calculate_action_value_probabilities(self, action_values):
        return softmax(action_values)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
