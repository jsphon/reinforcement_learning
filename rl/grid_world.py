import numpy as np
import pandas as pd

import copy
from rl.core import RLSystem, Model, State
from rl.reward_function import RewardFunction
from rl.value import StateValueFunction
from rl.policy import EpsilonGreedyPolicy

np.set_printoptions(precision=1)
np.set_printoptions(linewidth=200)
np.set_printoptions(suppress=True)

'''

Steps

1 . Generate Experience

    Generate n Episodes

        for i in range(n):

            Generate Sample Episode

    Simple Generate Sample Episode

        Initialise State

        while not terminal state:
            choose action
            take action

1 b.

    Generate root nodes
    For each root node:
        generate target value

2. Initialise StateAction Function (experience)
 - convert episode states to vectors
 - get rewards
 - fit the 1 step rewards against the target values so that we get decent starting values

3 . Fit StateAction Function (experience)

    Get States Array
        i.e. make a big array of states x num experience
    Get Target Values
        i.e. given the learner, such as sarsa or q, generate target values
    Fit(States, Target Value)


'''


class GridWorld(RLSystem):

    def __init__(self):
        super(GridWorld, self).__init__()
        self.policy = EpsilonGreedyPolicy()
        self.reward_function = GridRewardFunction()
        self.value_function = TabularValueFunction(self.policy)
        self.model = GridModel()

        self.num_actions = 4
        self.state_size = 64

        self.state_class = GridState

    def get_value_grid(self, state):
        values = np.ndarray((4, 4))
        for i in range(4):
            for j in range(4):
                state.player = (i, j)
                v = self.value_function.get_value(state.as_vector().reshape(1, 64))
                values[i, j] = v.max()
        return values

    def get_reward_grid(self, state):
        #state = GridState.new()
        rewards = np.ndarray((4, 4))
        for i in range(4):
            for j in range(4):
                state.player = (i, j)
                rewards[i, j] = self.reward_function.get_reward(None, None, state)
        return rewards


class GridRewardFunction(RewardFunction):

    def __call__(self, old_state, action, new_state):
        return self.get_reward(old_state, action, new_state)

    def get_reward(self, old_state, action, new_state):
        if new_state.player in ((0, 0), (3, 3)):
            return -0
        else:
            return -1


class GridModel(Model):

    def __init__(self):
        super(GridModel, self).__init__()
        self.get_new_state = new_fixed_grid_state
        self.num_actions = 4

    def apply_action(self, state, action):
        result = state.copy()
        if action == 0:
            self.move_up(result)
        elif action == 1:
            self.move_down(result)
        elif action == 2:
            self.move_left(result)
        elif action == 3:
            self.move_right(result)
        else:
            raise Exception
        return result

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
        new_position = (state.player[0], min(state.player[1] + 1, 3))
        state.update_player(new_position)

class TabularValueFunction(StateValueFunction):

    def __init__(self, policy):
        super(StateValueFunction, self).__init__(policy)
        self._values = np.zeros((4, 4), dtype=np.float)

    def __call__(self, state):
        return self._values[state.player[0], state.player[1]]


class GridState(State):

    def __init__(self, player):
        super(GridState, self).__init__()
        self.player = player
        self.size = 16
        self.vector_dtype = np.bool

    @staticmethod
    def all():
        states = []
        for i in range(4):
            for j in range(4):
                states.append(GridState((i, j)))
        return states

    def copy(self):
        return GridState(copy.copy(self.player))

    def as_vector(self):
        ''' Represent as a vector '''
        vec = np.zeros(self.size, dtype=self.vector_dtype)
        vec[4 * self.player[0] + self.player[1]] = True
        return vec

    def as_string(self):
        arr = self.as_2d_array()
        s = '\n'.join(''.join(row) for row in arr)
        s = s.replace(' ', '.')
        return s

    def as_2d_array(self):
        r = np.empty((4, 4), dtype='<U2')
        r[:] = ' '
        r[self.player[0], self.player[1]] = 'P'
        return r

    def update_player(self, player):
        self.player = player
        if player in ((0, 0), (3, 3)):
            self.is_terminal = True


def new_fixed_grid_state():
    """
    Create a new state with fixed values
    :return:
    """

    player = (0, 1)
    grid = GridState(player)
    return grid


def new_random_grid_state():
    """
    Create a new grid state with random values
    :return:
    """
    values = pd.DataFrame(np.random.randint(0, 3, (4, 2))).drop_duplicates().values
    if len(values) == 4:
        player = tuple(values[0])
        wall = tuple(values[1])
        pit = tuple(values[2])
        goal = tuple(values[3])

        grid = GridState(player, wall, pit, goal)
        return grid
    else:
        return new_random_grid_state()

