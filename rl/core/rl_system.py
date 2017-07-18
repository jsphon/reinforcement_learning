
class RLSystem(object):
    """
    From Sutton, a reinforcement system contains the following 4 components:
     - policy
     - reward function
     - value function
     - model
     """

    def __init__(self):
        self.policy = None
        self.reward_function = None
        self.action_value_function = None
        self.model = None
        self.get_new_state = None
        self.num_actions = 0
        self.state_size = 0
