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
        raise NotADirectoryError()

