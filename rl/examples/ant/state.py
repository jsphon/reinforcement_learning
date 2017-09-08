import numpy as np
from rl.core.state import IntExtState, State


class AntState(IntExtState):

    def __init__(self, internal_state=1, position=5):
        self.internal_state = internal_state
        self.external_state = ExternalState(position)
        self.num_homecomings = 0

    def __repr__(self):
        return str(self)

    def __str__(self):
        return '<AntState internal_state=%s, external_state.position=%s, num_homecomings=%s>'%(self.internal_state, self.external_state.position, self.num_homecomings)

    def copy(self):
        return AntState(internal_state=self.internal_state, position=self.external_state.position)


class ExternalState(State):

    def __init__(self, position=5):
        self.position = position

    def as_array(self):
        values = np.zeros(10, dtype=np.bool)
        values[self.position] = 1
        return values


if __name__ == '__main__':
    state = AntState()
    print(state.as_array().astype(np.int))
