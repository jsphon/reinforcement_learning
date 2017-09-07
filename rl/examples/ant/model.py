from rl.core.model import Model


class AntModel(Model):
    def __init__(self):
        super(Model, self).__init__()
        #self.num_actions =

    def apply_action(self, state, action):
        next_state = state.copy()
        if action == 0:
            self.move_up(next_state)
        elif action == 1:
            self.move_down(next_state)
        elif action == 2:
            self.move_left(next_state)
        elif action == 3:
            self.move_right(next_state)
        else:
            raise Exception('Unexpected action %s' % str(action))
        #print('walked to %s'%str(next_state.player))
        return next_state

    def move_up(self, state):
        new_position = (max(state.player[0] - 1, 0), state.player[1])
        state.update_player(new_position)

    def move_down(self, state):
        new_position = (min(state.player[0] + 1, 3), state.player[1])
        state.update_player(new_position)

    def move_left(self, state):
        new_position = (state.player[0], max(state.player[1] - 1, 0))
        state.update_player(new_position)

    def move_right(self, state):
        new_position = (state.player[0], min(state.player[1] + 1, 11))
        state.update_player(new_position)