import numpy as np
from rl.core.state import IntExtState, State


class AntState(State):
    def __init__(self, position=5):
        self._num_homecomings = 0
        self._is_terminal = False
        self.array_dtype = np.bool
        self._num_homecomings = 0
        self.size = 11
        self.position = position
        self.is_terminal = False

    def __repr__(self):
        return str(self)

    def __str__(self):
        return '<AntState external_state.position=%s, num_homecomings=%s>' % (
            self.position, self.num_homecomings)

    # def copy(self):
    #     return AntState(internal_state=self.internal_state, position=self.external_state.position)

    def as_array(self):
        values = np.zeros(self.size, dtype=np.bool)
        values[self.position] = 1
        values[10] = self._num_homecomings
        return values

    @property
    def num_homecomings(self):
        return self._num_homecomings

    @num_homecomings.setter
    def num_homecomings(self, value):
        self._num_homecomings = value
        if value == 5:
            self.is_terminal = True


#
# class ExternalState(State):
#
#     def __init__(self, position=5):
#         self.position = position
#         self.is_terminal=False
#         self.size = 11
#         self.array_dtype = np.bool
#         self._num_homecomings = 0
#
#     def as_array(self):
#         values = np.zeros(self.size, dtype=np.bool)
#         values[self.position] = 1
#         values[10] = self._num_homecomings
#         return values
#
#     @property
#     def num_homecomings(self):
#         return self._num_homecomings
#
#     @num_homecomings.setter
#     def num_homecomings(self, value):
#         self._num_homecomings = value
#         if value == 5:
#             self.is_terminal=True


if __name__ == '__main__':
    state = AntState()
    print(state.as_array().astype(np.int))
