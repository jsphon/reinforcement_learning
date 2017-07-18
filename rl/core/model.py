

class Model(object):

    def apply_action(self, state, action):
        ''' Might predict the next state and reward'''
        raise NotImplemented()