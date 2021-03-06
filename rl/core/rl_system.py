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
            state_size=0,
    ):
        self.policy = policy
        self.reward_function = reward_function
        self.action_value_function = action_value_function
        self.model = model
        self.get_new_state = None
        self.state_size = state_size

    def choose_action(self, state):
        probabilities = self.calculate_action_value_probabilities(state)
        action = np.random.choice(len(probabilities), p=probabilities)
        return action

    def calculate_action_value_probabilities(self, state):
        action_values = self.action_value_function(state)
        return self.policy.calculate_action_value_probabilities(action_values)


class StateMachineSystem(RLSystem):

    def __init__(self, *args, **kwargs):
        super(StateMachineSystem, self).__init__(*args, **kwargs)
        self.num_internal_states = 0
        self.num_actions = []
        self.external_states_meta = []
