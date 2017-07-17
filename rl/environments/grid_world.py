import pandas as pd

import copy
from rl.core import RLSystem, Model, State
from rl.reward_function import RewardFunction
from rl.value import ActionValueFunction
from rl.policy import EpsilonGreedyPolicy

from keras.layers.core import Dense, Activation
from keras.models import Sequential
from keras.optimizers import RMSprop
import numpy as np

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
        self.action_value_function = GridActionValueFunction()
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


class GridActionValueFunction(ActionValueFunction):

    def __init__(self):
        model = Sequential()
        model.add(Dense(164, kernel_initializer='lecun_uniform', input_shape=(16,)))
        model.add(Activation('relu'))
        # model.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a try?

        model.add(Dense(150, kernel_initializer='lecun_uniform'))
        model.add(Activation('relu'))
        # model.add(Dropout(0.2))

        model.add(Dense(4, kernel_initializer='lecun_uniform'))
        model.add(Activation('linear'))  # linear output so we can have range of real-valued outputs

        rms = RMSprop()
        model.compile(loss='mse', optimizer=rms)
        self.model = model

    def __call__(self, state):
        """

        :param state:
        :return: np.ndarray(num_actions)
        """
        arr = state.as_array()
        return self.model.predict(arr)

    def on_list(self, states):

        state_size = states[0].size
        num_states = len(states)
        dtype = states[0].array_dtype
        arr = np.ndarray((num_states, state_size), dtype=dtype)
        for i in range(num_states):
            arr[i, :] = states[i].as_array()
        return self.model.predict(arr)

    def fit(self, x, y, **kwargs):
        self.model.fit(x, y, **kwargs)


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
            raise Exception('Unexpected action %s' % str(action))
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


class GridState(State):

    def __init__(self, player):
        super(GridState, self).__init__()
        self.player = player
        self.size = 16
        self.array_dtype = np.bool

    def __repr__(self):
        return '<GridState player=%s>' % str(self.player)

    @staticmethod
    def enumerate(self):

        states = []
        for i in range(4):
            for j in range(4):
                player = (i, j)
                state = GridState(player)
                if not state.is_terminal:
                    states.append(state)
        return states

    @staticmethod
    def all():
        """
        Get all posible states
        :return:
        """
        states = []
        for i in range(4):
            for j in range(4):
                states.append(GridState((i, j)))
        return states

    def copy(self):
        return GridState(copy.copy(self.player))

    def as_array(self):
        ''' Represent as an array '''
        vec = np.zeros((1, self.size), dtype=self.array_dtype)
        vec[0, 4 * self.player[0] + self.player[1]] = True
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
