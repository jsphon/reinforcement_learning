import numpy as np


class State(object):
    """
    States need to be translatable to numpy arrays for training.
    However, other representations can be more convenient to work with when we aren't
    training.
    """

    def __init__(self, meta=None):

        self.meta = meta or StateMeta()

        # State Data
        self.is_terminal = False

    @property
    def size(self):
        return self.meta.size

    @size.setter
    def size(self, value):
        self.meta.size = value

    @property
    def array_dtype(self):
        return self.meta.array_dtype

    @array_dtype.setter
    def array_dtype(self, value):
        self.meta.array_dtype = value

    @property
    def num_actions(self):
        return self.meta.num_actions

    @num_actions.setter
    def num_actions(self, value):
        self.meta.num_actions=value

    def as_array(self):
        """
        For input to the reward function fitting
        :return:
        """
        raise NotImplemented()


class StateMeta(object):

    def __init__(self):
        self.size = 0
        self.array_dtype = np.float
        self.num_actions = 2


class StateList(object):

    def __init__(self, states=None, meta=None):
        self.meta = meta or states[0].meta if states else StateMeta()
        self.values = states or []

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return self.values.__iter__()

    @property
    def num_actions(self):
        return self.meta.num_actions

    def as_array(self):
        state_size = self[0].size
        num_training_states = len(self)
        dtype = self[0].array_dtype
        result = np.ndarray((num_training_states, state_size), dtype=dtype)
        for i in range(num_training_states):
            result[i, :] = self[i].as_array()
        return result


class IntExtState(object):
    """
    Represents a State which is also the state of a finite state machine
    """

    def __init__(self, internal_state, external_state):

        self.external_state = external_state
        self.internal_state = internal_state

    @property
    def meta(self):
        return self.external_state.meta

    @property
    def num_actions(self):
        return self.external_state.num_actions

    def copy(self):
        return IntExtState(self.internal_state, self.external_state)

    @property
    def size(self):
        return self.external_state.size

    @property
    def array_dtype(self):
        return self.external_state.array_dtype

    def as_array(self):
        return self.external_state.as_array()


