import copy

import numpy as np

from rl.core.model import Model
from rl.core.policy import EpsilonGreedyPolicy
from rl.core.reward_function import RewardFunction
from rl.core.state import State
from rl.core.value_function import ActionValueFunction
from rl.environments.grid_world.simple_grid_world import SimpleGridWorld

np.set_printoptions(precision=1)
np.set_printoptions(linewidth=200)
np.set_printoptions(suppress=True)


class CliffWorld(SimpleGridWorld):
    def __init__(self):
        super(CliffWorld, self).__init__()
        self.policy = EpsilonGreedyPolicy(self)
        self.reward_function = GridRewardFunction()
        self.action_value_function = GridActionValueFunction()
        self.model = GridModel()

        self.shape = (4, 12)

        self.state_class = GridState

    def get_value_grid(self):
        values = np.ndarray(self.shape)

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                state = self.state_class(player=(i, j))
                if self.model.is_terminal(state):
                    values[i, j] = np.nan
                else:
                    v = self.action_value_function(state)
                    values[i, j] = v.max()
        return values

    def is_terminal_grid(self):
        values = np.ndarray(self.shape, dtype=np.bool)

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                state = self.state_class(player=(i, j))
                values[i, j] = self.model.is_terminal(state)
        return values

    def get_greedy_action_grid_string(self):

        action_to_char = {
            0: '^', 1: 'v', 2: '<', 3: '>', -1: '.'
        }

        s_actions = np.ndarray((self.shape[0], self.shape[1]), dtype=np.dtype('U1'))
        actions = self.get_greedy_action_grid()
        for i in range(actions.shape[0]):
            for j in range(actions.shape[1]):
                s_actions[i, j] = action_to_char[actions[i, j]]
        rows = [''.join(row) for row in s_actions]
        return '\n'.join(rows)

    def get_reward_grid(self):
        rewards = np.ndarray(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                state = GridState(player=(i, j))
                rewards[i, j] = self.reward_function.get_reward(None, None, state)
        return rewards


class GridActionValueFunction(ActionValueFunction):

    def __init__(self):
        from collections import defaultdict
        self.values = defaultdict(self.new_value)
        self.learning_rate = 0.1

    def new_value(self):
        return np.zeros(4)

    def __call__(self, state):
        """

        :param state:
        :return: np.ndarray(num_actions)
        """
        key = str(state)
        return self.values[key]

    def vectorized_fit(self, states, targets, **kwargs):
        """

        Args:
            states: list of states
            targets: (num_states x num_actions) array of target values
            **kwargs:

        Returns:

        """
        for i in range(len(states)):
            state = states[i]
            key = str(state)
            _y = targets[i]
            current_value = self.values[key]
            diff = (_y - current_value)
            target = current_value + self.learning_rate * diff
            self.values[key] = target
            #print("Updated %s's value from %s to %s" % (key, str(current_value), target))

    def scalar_fit(self, states, actions, targets):
        """

        Args:
            states: list of states
            actions: list of actions
            targets: list of targets

        Returns:

        """
        for i in range(len(states)):
            state = states[i]
            key = str(state)
            current_value = self.values[key][actions[i]]
            diff = (targets[i] - current_value)
            target = current_value + self.learning_rate * diff
            self.values[key][actions[i]] = target
            #print("Updated Q(%s, %s) from %s to %s" % (key, actions[i], str(current_value), target))


class GridRewardFunction(RewardFunction):
    def __call__(self, old_state, action, new_state):
        return self.get_reward(old_state, action, new_state)

    def get_reward(self, old_state, action, new_state):
        if walked_off_cliff(new_state):
            return -100
        else:
            return -1


def walked_off_cliff(new_state):
    return (new_state.player[0] == 0) \
           and (0 < new_state.player[1] < 11)


class GridModel(Model):
    def __init__(self):
        super(GridModel, self).__init__()
        self.get_new_state = new_fixed_grid_state
        self.num_actions = 4

    def apply_action(self, state, action):
        next_state = GridState(player=state.player)
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

    def is_terminal(self, state):
        return (state.player[0] == 0) and (state.player[1] > 0)


class GridState(State):

    size = 4 * 12
    shape = (4, 12)
    array_dtype = np.bool

    def __init__(self, player):
        super(GridState, self).__init__()
        self.player = None
        self.array_dtype = np.bool
        self.player = player

    def __repr__(self):
        return '<GridState player=%s>' % str(self.player)

    @classmethod
    def all(cls):

        states = []
        for i in range(cls.shape[0]):
            for j in range(cls.shape[1]):
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
            for j in range(12):
                states.append(GridState((i, j)))
        return states

    def copy(self):
        r = GridState(copy.copy(self.player), visited_states=copy.copy(self.visited_states))
        return r

    def as_array(self):
        ''' Represent as an array '''
        vec = np.zeros((1, self.size), dtype=self.array_dtype)
        vec[0, 12 * self.player[0] + self.player[1]] = True
        return vec

    def as_string(self):
        arr = self.as_2d_array()
        s = '\n'.join(''.join(row) for row in arr)
        s = s.replace(' ', '.')
        return s

    def as_2d_array(self):
        r = np.empty((4, 12), dtype='<U2')
        r[:] = ' '
        r[self.player[0], self.player[1]] = 'P'
        return r

    def update_player(self, player):
        self.player = player


def new_fixed_grid_state():
    """
    Create a new state with fixed values
    :return:
    """

    player = (0, 0)
    grid = GridState(player)
    return grid
