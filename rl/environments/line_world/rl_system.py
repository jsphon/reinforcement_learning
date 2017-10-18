
import numpy as np
from rl.environments.line_world.model import AntModel
from rl.environments.line_world.reward_function import AntRewardFunction
from rl.environments.line_world.state import AntState

from rl.core.policy import EpsilonGreedyPolicy
from rl.core.rl_system import RLSystem
from rl.core.state import IntExtState
from rl.environments.line_world.value_function import AntActionValueFunction
from rl.environments.line_world.constants import ACTION_STRINGS


class AntWorld(RLSystem):
    def __init__(self):
        super(RLSystem, self).__init__()

        self.reward_function = AntRewardFunction()
        self.model = AntModel()
        self.action_value_function = AntActionValueFunction()
        self.policy = EpsilonGreedyPolicy(epsilon=0)
        self.num_actions = (2, 2)
        self.num_internal_states = 2

    def calculate_action_values(self):

        result = []
        for internal_state in (0, 1):
            action_values = []
            for _position in range(10):
                _external_state = AntState(position=_position)
                _state = IntExtState(internal_state, _external_state)
                avs = self.action_value_function(_state)
                action_values.append(avs)
            result.append(np.r_[action_values])
        return result


def calculate_greedy_actions(ant_world):
    rows = []
    null_cells = [(0, 2), (1, 8)]
    for internal_state in (0, 1):
        row = []
        for _position in range(10):
            _external_state = AntState(position=_position)
            _state = IntExtState(internal_state, _external_state)
            _action = ant_world.choose_action(_state)
            if (internal_state, _position) in null_cells:
                row.append('x')
            else:
                row.append(ACTION_STRINGS[_action])
        rows.append(''.join(row))

    row0 = 'FINDING HOME : %s' % rows[0]
    row1 = 'FINDING FOOD : %s' % rows[1]
    return row0 + '\n' + row1


