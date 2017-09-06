import numpy as np


class State(object):
    """
    States need to be translatable to numpy arrays for training.
    However, other representations can be more convenient to work with when we aren't
    training.
    """

    def __init__(self):
        self.is_terminal = False
        self.size = 0
        self.array_dtype = np.float

    def as_array(self):
        """
        For input to the reward function fitting
        :return:
        """
        raise NotImplemented()


class IntExtState(object):
    """
    Represents a State which is also the state of a finite state machine
    """

    def __init__(self, internal_state, external_state):
        """

        Args:
            num_states: int
            num_actions: list of ints, one for each state
        """
        super(IntExtState, self).__init__()

        self.external_state = external_state
        self.internal_state = internal_state

    @property
    def is_terminal(self):
        return self.external_state.is_terminal

    @property
    def size(self):
        return self.external_state.size

    @property
    def array_dtype(self):
        return self.external_state.array_dtype

    def as_array(self):
        return self.external_state.as_array()
