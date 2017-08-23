import numpy as np

from rl.core.policy import EpsilonGreedyPolicy
from rl.core.rl_system import RLSystem
from rl.core.value_function import ActionValueFunction


class BaseGridWorld(RLSystem):
    def __init__(self):
        super(BaseGridWorld, self).__init__()
        self.policy = EpsilonGreedyPolicy(self)

        self.num_actions = 4
        self.state_class = None
        self.shape = None

        self.action_value_function = None


class TabularGridActionValueFunction(ActionValueFunction):

    def __init__(self, num_actions=4):
        from collections import defaultdict
        self.values = defaultdict(self.new_value)
        self.learning_rate = 0.1
        self.num_actions = num_actions

    def new_value(self):
        return np.zeros(self.num_actions)

    def __call__(self, state):
        """

        :param state:
        :return: np.ndarray(num_actions)
        """
        key = str(state)
        return self.values[key].copy()

    def on_list(self, states):
        """

        Args:
            states: a list of states

        Returns:
            np.array of shape (num_states, num_actions)
        """
        return [self(state) for state in states]

    def vectorized_fit(self, states, targets, **kwargs):
        """

        Args:
            states: list of states
            targets: (num_states x num_actions) array of target values
            **kwargs:

        Returns:

        """
        for i in range(len(states)):
            state = states[i]
            key = str(state)
            _y = targets[i]
            current_value = self.values[key]
            diff = (_y - current_value)
            target = current_value + self.learning_rate * diff
            self.values[key] = target
            print("Updated %s's value from %s to %s" % (key, str(current_value), target))

    def scalar_fit(self, states, actions, targets):
        """

        Args:
            states: list of states
            actions: list of actions
            targets: list of targets

        Returns:

        """
        for i in range(len(states)):
            state = states[i]
            key = str(state)
            current_value = self.values[key][actions[i]]
            diff = (targets[i] - current_value)
            target = current_value + self.learning_rate * diff
            self.values[key][actions[i]] = target
            print("Updated Q(%s, %s) from %s to %s" % (key, actions[i], str(current_value), target))
