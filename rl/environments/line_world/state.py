import numpy as np
from rl.core.state import State, StateMeta


class AntStateMeta(StateMeta):

    def __init__(self):
        super(AntStateMeta, self).__init__(
            size = 11,
            num_actions = 2,
            array_dtype = np.float
        )


class AntState(State):

    def __init__(
            self,
            position=5,
            num_homecomings=0):
        self._num_homecomings = num_homecomings
        self.position = position

        self.meta = AntStateMeta()

    def __repr__(self):
        return str(self)

    def __str__(self):
        return '<AntState position=%s, num_homecomings=%s>' % (
            self.position, self.num_homecomings)

    def copy(self):
        return AntState(
            position=self.position,
            num_homecomings=self._num_homecomings
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


if __name__ == '__main__':
    state = AntState()
    print(state.as_array().astype(np.int))
