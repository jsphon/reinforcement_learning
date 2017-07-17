
class ValueFunction(object):

    def __init__(self, policy):
        self.policy = policy


class StateValueFunction(ValueFunction):
    """
    Known as v_{\pi}(s) in the literature.
    """
    def __call__(self, state):
        pass


class ActionValueFunction(ValueFunction):
    """
    Known as q(s, a) in the literature.
    """

    def __call__(self, state):
        """

        :param state:
        :return: np.ndarray(num_actions)
        """
        pass