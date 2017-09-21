import numpy as np
from rl.core.state import State


class AntState(State):

    def __init__(
            self,
            position=5,
            num_homecomings=0,
            max_home_comings=1):
        self._is_terminal = False
        self.array_dtype = np.bool
        self._num_homecomings = num_homecomings
        self.size = 11
        self.position = position
        self.is_terminal = False
        self.max_home_comings = max_home_comings

    def __repr__(self):
        return str(self)

    def __str__(self):
        return '<AntState position=%s, num_homecomings=%s>' % (
            self.position, self.num_homecomings)

    def copy(self):
        return AntState(
            position=self.position,
            num_homecomings=self._num_homecomings,
            max_home_comings=self.max_home_comings
        )

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
        if value == self.max_home_comings:
            self.is_terminal = True


if __name__ == '__main__':
    state = AntState()
    print(state.as_array().astype(np.int))
