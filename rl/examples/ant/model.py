from rl.core.model import Model
import numpy as np

FINDING_HOME = 0
FINDING_FOOD = 1

MOVE_LEFT = 0
MOVE_RIGHT = 1

HOME_POSITION = 2
FOOD_POSITION = 8


class AntModel(Model):
    def __init__(self):
        super(Model, self).__init__()

    def apply_action(self, state, action):

        internal_state = state.internal_state

        if internal_state == FINDING_HOME:
            return self.apply_finding_home_action(state, action)
        elif internal_state == FINDING_FOOD:
            return self.apply_finding_food_action(state, action)
        else:
            raise ValueError('Action should be 0 or 1')

    def apply_finding_food_action(self, state, action):

        new_state = self.apply_movement(state, action)
        if new_state.external_state.position == FOOD_POSITION:
            print('Changing state to finding home')
            new_state.internal_state = FINDING_HOME

        return new_state

    def apply_finding_home_action(self, state, action):

        new_state = self.apply_movement(state, action)
        if new_state.external_state.position == HOME_POSITION:
            print('changing state to finding food')
            new_state.internal_state = FINDING_FOOD
            new_state.num_homecomings += 1

        return new_state

    def apply_movement(self, state, action):
        new_state = state.copy()
        external_state = state.external_state

        if action == MOVE_RIGHT:
            new_state.external_state.position = min(9, external_state.position + 1)
        elif action == MOVE_LEFT:
            new_state.external_state.position = max(0, external_state.position - 1)

        return new_state


if __name__=='__main__':

    from rl.examples.ant.state import AntState
    model = AntModel()
    ant_state = AntState()

    print(ant_state)