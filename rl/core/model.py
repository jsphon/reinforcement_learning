

class Model(object):

    def apply_action(self, state, action):
        ''' Might predict the next state and reward'''
        raise NotImplemented()

    def is_terminal(self, state):
        ''' Return True if the state is terminal'''
        raise NotImplemented()