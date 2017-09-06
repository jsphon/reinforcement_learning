
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


class StateMachineActionValueFunction(ValueFunction):
    """
    An action value function for state machine.

    """

    def __call__(self, int_ext_state):
        pass


class NeuralNetStateMachineActionValueFunction(StateMachineActionValueFunction):

    """
    For state machines
    """

    def __init__(self):
        super(NeuralNetStateMachineActionValueFunction, self).__init__()
        # One model for each state
        self.state_models = []

    def __call__(self, int_ext_state):
        internal_state = int_ext_state.internal_state
        arr = int_ext_state.external_state.as_array()
        return self.state_models[internal_state].predict(arr).ravel()
