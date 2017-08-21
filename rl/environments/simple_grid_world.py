import copy

import numpy as np
from keras.layers.core import Dense, Activation
from keras.models import Sequential
from keras.optimizers import RMSprop

from rl.core.model import Model
from rl.core.policy import EpsilonGreedyPolicy
from rl.core.reward_function import RewardFunction
from rl.core.state import State
from rl.core.value_function import ActionValueFunction
from rl.environments.base_grid_world import BaseGridWorld
from rl.environments.base_grid_world import TabularGridActionValueFunction


class SimpleGridWorld(BaseGridWorld):

    def __init__(self):
        super(SimpleGridWorld, self).__init__()

        self.reward_function = SimpleGridWorldRewardFunction()
        self.model = SimpleGridModel()

        self.state_class = SimpleGridState
        self.shape = (4, 4)

        self.action_value_function = TabularGridActionValueFunction(self.num_actions)

    def get_value_grid(self):
        values = np.ndarray(self.shape)

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                state = self.state_class(player=(i, j))
                if state.is_terminal:
                    values[i, j] = np.nan
                else:
                    v = self.action_value_function(state)
                    values[i, j] = v.max()
        return values

    def get_greedy_action_grid_string(self):

        action_to_char = {
            0: '^', 1: 'v', 2: '<', 3: '>', -1: '.'
        }

        s_actions = np.ndarray(self.shape, dtype=np.dtype('U1'))
        actions = self.get_greedy_action_grid()
        for i in range(actions.shape[0]):
            for j in range(actions.shape[1]):
                s_actions[i, j] = action_to_char[actions[i, j]]
        rows = [''.join(row) for row in s_actions]
        return '\n'.join(rows)

    def get_greedy_action_grid(self):

        actions = np.ndarray(self.shape, dtype=np.int8)
        policy = EpsilonGreedyPolicy(rl_system=self, epsilon=0)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                state = self.state_class(player=(i, j))
                if state.is_terminal:
                    actions[i, j] = -1
                else:
                    actions[i, j] = policy.choose_action(state)

        return actions

    def get_reward_grid(self):
        rewards = np.ndarray(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                state = SimpleGridState(player=(i, j))
                rewards[i, j] = self.reward_function.get_reward(None, None, state)
        return rewards

#
# class GridActionValueFunction(ActionValueFunction):
#
#     def __init__(self, shape, num_actions):
#
#         input_size = shape[0] * shape[1]
#
#         model = Sequential()
#         model.add(Dense(164, kernel_initializer='lecun_uniform', input_shape=(input_size,)))
#         model.add(Activation('relu'))
#         # model.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a try?
#
#         model.add(Dense(150, kernel_initializer='lecun_uniform'))
#         model.add(Activation('relu'))
#         # model.add(Dropout(0.2))
#
#         model.add(Dense(num_actions, kernel_initializer='lecun_uniform'))
#         model.add(Activation('linear'))  # linear output so we can have range of real-valued outputs
#
#         rms = RMSprop()
#         model.compile(loss='mse', optimizer=rms)
#         self.model = model
#
#     def __call__(self, state):
#         """
#
#         :param state:
#         :return: np.ndarray(num_actions)
#         """
#         arr = state.as_array()
#         return self.model.predict(arr).ravel()
#
#     def on_list(self, states):
#         """
#
#         :param states:
#         :return: np.ndarray(len(states), num_actions)
#         """
#
#         state_size = states[0].size
#         num_states = len(states)
#         dtype = states[0].array_dtype
#         arr = np.ndarray((num_states, state_size), dtype=dtype)
#         for i in range(num_states):
#             arr[i, :] = states[i].as_array()
#         return self.model.predict(arr)
#
#     def vectorized_fit(self, states, targets, **kwargs):
#
#         x = states.as_array()
#         self.model.fit(x, targets, **kwargs)
#
#     def scalar_fit(self, states, actions, rewards, **kwargs):
#         pass


class SimpleGridWorldRewardFunction(RewardFunction):
    def __call__(self, old_state, action, new_state):
        return self.get_reward(old_state, action, new_state)

    def get_reward(self, old_state, action, new_state):
        if new_state.player in ((0, 0), (3, 3)):
            return 0
        else:
            return -1


class SimpleGridModel(Model):

    def __init__(self):
        super(SimpleGridModel, self).__init__()
        self.get_new_state = new_random_simple_grid_state

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


class SimpleGridState(State):
    def __init__(self, player):
        super(SimpleGridState, self).__init__()
        self.player = None
        self.size = 16
        self.array_dtype = np.bool

        self.update_player(player)

    def __repr__(self):
        return '<SimpleGridState player=%s>' % str(self.player)

    def copy(self):
        return SimpleGridState(copy.copy(self.player))

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
    grid = SimpleGridState(player)
    return grid


def new_random_simple_grid_state():
    """
    Create a new grid state with random values
    :return:
    """
    player = np.random.randint(0, 4, 2).tolist()
    grid = SimpleGridState(player=player)
    return grid
