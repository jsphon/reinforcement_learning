
from rl.core.policy import EpsilonGreedyPolicy
from rl.core.rl_system import RLSystem


class BaseGridWorld(RLSystem):

    def __init__(self):
        super(BaseGridWorld, self).__init__()
        self.policy = EpsilonGreedyPolicy(self)

        self.num_actions = 4
        self.state_class = None
        self.shape = None

        self.action_value_function = None
