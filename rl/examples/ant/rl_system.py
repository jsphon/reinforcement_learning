
import numpy as np
from rl.core.rl_system import RLSystem

from rl.examples.ant.reward_function import AntRewardFunction
from rl.examples.ant.model import AntModel
from rl.examples.ant.value_function import AntActionValueFunction
from rl.core.policy import EpsilonGreedyPolicy# SoftmaxPolicy


class AntWorld(RLSystem):
    def __init__(self):
        super(RLSystem, self).__init__()

        self.reward_function = AntRewardFunction()
        self.model = AntModel()
        self.action_value_function = AntActionValueFunction()
        #self.policy = SoftmaxPolicy()
        self.policy = EpsilonGreedyPolicy(epsilon=0)
        self.num_actions = (2, 2)
        self.num_internal_states = 2

        # self.state_class = SimpleGridState
        # self.shape = (4, 4)

        # self.action_value_function = TabularGridActionValueFunction(self.num_actions)

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
    for internal_state in (0, 1):
        row = []
        for _position in range(10):
            _external_state = AntState(position=_position)
            _state = IntExtState(internal_state, _external_state)
            _action = ant_world.choose_action(_state)
            row.append(str(_action))
        rows.append(''.join(row))

    _greedy_actions = '\n'.join(rows)
    return _greedy_actions


if __name__ == '__main__':

    from rl.examples.ant.state import AntState
    from rl.core.state import IntExtState

    external_state = AntState()
    int_ext_state = IntExtState(0, external_state)
    world = AntWorld()

    world.choose_action(int_ext_state)

    from rl.core.learning.learner import build_vectorized_state_machine_q_learner

    learner = build_vectorized_state_machine_q_learner(world)

    from rl.core.experience import StatesList

    states_list = StatesList()
    for position in range(10):
        state = AntState(position=position)
        states_list.append(state)

    for _ in range(1000): learner.learn(states_list)

    state = IntExtState(1, external_state)

    greedy_actions = calculate_greedy_actions(world)

    print(greedy_actions)

    state = IntExtState(0, AntState(1))

    action_values = world.calculate_action_values()
    print(action_values)
