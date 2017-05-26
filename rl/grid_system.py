import random

from keras.layers.core import Dense, Activation
from keras.models import Sequential
from keras.optimizers import RMSprop
import numpy as np

from rl.core import RLSystem, Policy, RewardFunction, ValueFunction, Model

np.set_printoptions(precision=1)
np.set_printoptions(linewidth=200)
np.set_printoptions(suppress=True)


class GridSystem(RLSystem):

    def __init__(self):
        super(GridSystem, self).__init__()
        self.policy = GridPolicy()
        self.reward_function = GridRewardFunction()
        self.value_function = GridValueFunction()
        self.model = GridModel()


        self.num_actions = 4
        self.state_size = 64

    def get_value_grid(self):

        state = GridState.new()
        values = np.ndarray((4, 4))
        for i in range(4):
            for j in range(4):
                state.player = (i, j)
                v = self.value_function.get_value(state.as_vector().reshape(1, 64))
                values[i, j] = v.max()
        return values


    def get_reward_grid(self):
        state = GridState.new()
        rewards = np.ndarray((4, 4))
        for i in range(4):
            for j in range(4):
                state.player = (i, j)
                rewards[i, j] = self.reward_function.get_reward(None, None, state)
        return rewards


class GridPolicy(Policy):

    def __init__(self):
        self.epsilon=0.1

    def get_action(self, action_values):
        if (random.random() < self.epsilon):  # choose random action
            action = np.random.randint(0, 4)
        else:  # choose best action from Q(s,a) values
            action = (np.argmax(action_values))
        return action


class GridRewardFunction(RewardFunction):

    def get_reward(self, old_state, action, new_state):

        if new_state.player == new_state.pit:
            return -10
        elif new_state.player == new_state.goal:
            return 10
        return -1


class GridValueFunction(ValueFunction):

    def __init__(self):
        model = Sequential()
        model.add(Dense(164, kernel_initializer='lecun_uniform', input_shape=(64,)))
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

    def get_value(self, state):
        return self.model.predict(state)

    def fit(self, x, y, **kwargs):
        self.model.fit(x, y, **kwargs)


class GridState(object):

    def __init__(self, player, wall, pit, goal):
        self.player = player
        self.wall = wall
        self.pit = pit
        self.goal = goal
        self.size = 64
        self.is_terminal=False

    @staticmethod
    def new():
        player = (0, 1)
        wall = (2, 2)
        pit = (1, 1)
        goal = (3, 3)
        grid = GridState(player, wall, pit, goal)
        return grid

    def copy(self):
        return GridState(self.player, self.wall, self.pit, self.goal)

    def as_vector(self):
        ''' Represent as a vector '''
        vec = np.zeros(64, dtype=np.bool)
        vec[4 * self.player[0] + self.player[1]] = True
        vec[16 + 4 * self.wall[0] + self.wall[1]] = True
        vec[32 + 4 * self.pit[0] + self.pit[1]] = True
        vec[48 + 4 * self.goal[0] + self.goal[1]] = True
        return vec

    def as_2d_array(self):
        r = np.empty((4, 4), dtype='<U2')
        r[:] = ' '
        r[self.player[0], self.player[1]] = 'P'
        r[self.wall[0], self.wall[1]] = 'W'
        r[self.pit[0], self.pit[1]] = '-'
        r[self.goal[0], self.goal[1]] = '+'
        return r

    def apply_action(self, action):
        if action == 0:
            return self.move_up()
        elif action == 1:
            return self.move_down()
        elif action == 2:
            return self.move_left()
        elif action == 3:
            return self.move_right()
        else:
            raise Exception

    def move_up(self):
        new_position = (max(self.player[0] - 1, 0), self.player[1])
        if new_position != self.wall:
            self._update_player(new_position)

    def move_down(self):
        new_position = (min(self.player[0] + 1, 3), self.player[1])
        if new_position != self.wall:
            self._update_player(new_position)

    def move_left(self):
        new_position = (self.player[0], max(self.player[1] - 1, 0))
        if new_position != self.wall:
            self._update_player(new_position)

    def move_right(self):
        new_position = (self.player[0], min(self.player[1] + 1, 3))
        if new_position != self.wall:
            self._update_player(new_position)

    def _update_player(self, player):
        self.player=player
        if player in (self.pit, self.goal):
            self.is_terminal=True


class GridModel(Model):

    def __init__(self):
        super(GridModel, self).__init__()
        self.reward_function = GridRewardFunction()
        self.value_function = GridValueFunction()
        self.get_new_state = GridState.new
        self.num_actions = 4

    def apply_action(self, state, action):
        new_state = state.copy()
        new_state.apply_action(action)
        return new_state#, reward



if __name__=='__main__':
    grid_sys = GridSystem()
    grid_sys.num_actions = 4
    grid_sys.state_size = 64

    grid_sys.initialise_value_function()

    print('initial value function')
    print(grid_sys.get_value_grid())

    gamma = 0.9
    for i in range(10):
        states, action_rewards, new_states, new_states_terminal = grid_sys.generate_experience(num_epochs=10)

        N = len(new_states)
        new_values = grid_sys.generate_action_target_values(new_states)
        new_values[new_states_terminal] = 0

        targets = action_rewards + gamma * new_values
        grid_sys.value_function.fit(states, targets, verbose=0)

        # print('targets, action_rewards, new_values for epoch %i' % i)
        # print(str(np.c_[targets, action_rewards, new_values]))
        #
        print('Values after epoch %s'%i)
        print(grid_sys.get_value_grid())

    np.set_printoptions(precision=1)
    #print(grid_sys.value_function.get_value(states))


    # Display values of each location

    print('values')
    values = grid_sys.get_value_grid()
    print(values)

    print('2d grid')
    state = GridState.new()
    print(state.as_2d_array())

    print('Reward Grid')
    print(grid_sys.get_reward_grid())
