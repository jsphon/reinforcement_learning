import numpy as np
from rl.core.state import IntExtState, State


class AntState(IntExtState):
    def __init__(self):
        self.internal_state = 0
        self.external_state = ExternalState()


class ExternalState(State):
    def __init__(self, pos=5):
        self.pos = pos
        self.values = np.zeros(10, dtype=np.bool)
        self.values[pos] = 1

    def as_array(self):
        return self.values


if __name__ == '__main__':
    state = AntState()
    print(state.as_array().astype(np.int))
