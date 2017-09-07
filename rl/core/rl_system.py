import numpy as np


class RLSystem(object):
    """
    From Sutton, a reinforcement system contains the following 4 components:
     - policy
     - reward function
     - value function
     - model
     """

    def __init__(
            self,
            policy=None,
            reward_function=None,
            action_value_function=None,
            model=None,
            num_actions=0,
            state_size=0,
    ):
        self.policy = policy
        self.reward_function = reward_function
        self.action_value_function = action_value_function
        self.model = model
        self.get_new_state = None
        self.num_actions = num_actions
        self.state_size = state_size

    def choose_action(self, state):
        probabilities = self.calculate_state_probabilities(state)
        action = np.random.choice(len(probabilities), p=probabilities)
        return action

    def calculate_state_probabilities(self, state):
        action_values = self.action_value_function(state)
        return self.policy.calculate_action_value_probabilities(action_values)
