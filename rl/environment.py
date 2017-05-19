import numba
import numpy as np


@numba.jit(nopython=True)
def rand_pair(s, e):
    return np.random.randint(s, e), np.random.randint(s, e)


def init_grid():
    player = (0, 1)
    wall = (2, 2)
    pit = (1, 1)
    goal = (3, 3)
    grid = Grid(player, wall, pit, goal)
    return grid


class Grid(object):
    def __init__(self, player, wall, pit, goal):
        self.player = player
        self.wall = wall
        self.pit = pit
        self.goal = goal

    def as_one_hot(self):
        ''' Represent as a 1-hot vector '''
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

    def get_reward(self):
        return self.get_reward_if_player(self.player)

    def get_reward_if_player(self, player):
        if player == self.pit:
            return -10
        elif player == self.goal:
            return 10
        return -1

    def move_up(self):
        self.player = self.if_move_up()

    def if_move_up(self):
        new_position = (max(self.player[0] - 1, 0), self.player[1])
        if new_position != self.wall:
            return new_position
        else:
            return self.player

    def move_down(self):
        self.player = self.if_move_down()

    def if_move_down(self):
        new_position = (min(self.player[0] + 1, 3), self.player[1])
        if new_position != self.wall:
            return new_position
        else:
            return self.player

    def move_left(self):
        self.player = self.if_move_left()

    def if_move_left(self):
        new_position = (self.player[0], max(self.player[1] - 1, 0))
        if new_position != self.wall:
            return new_position
        else:
            return self.player

    def move_right(self):
        self.player = self.if_move_right()

    def if_move_right(self):
        new_position = (self.player[0], min(self.player[1] + 1, 3))
        if new_position != self.wall:
            return new_position
        else:
            return self.player


def randPair(s, e):
    return np.random.randint(s, e), np.random.randint(s, e)


# finds an array in the "depth" dimension of the grid
def findLoc(state, obj):
    for i in range(0, 4):
        for j in range(0, 4):
            if (state[i, j] == obj).all():
                return i, j


# Initialize stationary grid, all items are placed deterministically
def initGrid():
    state = np.zeros((4, 4, 4))
    # place player
    state[0, 1] = np.array([0, 0, 0, 1])
    # place wall
    state[2, 2] = np.array([0, 0, 1, 0])
    # place pit
    state[1, 1] = np.array([0, 1, 0, 0])
    # place goal
    state[3, 3] = np.array([1, 0, 0, 0])

    return state


# Initialize player in random location, but keep wall, goal and pit stationary
def initGridPlayer():
    state = np.zeros((4, 4, 4))
    # place player
    state[randPair(0, 4)] = np.array([0, 0, 0, 1])
    # place wall
    state[2, 2] = np.array([0, 0, 1, 0])
    # place pit
    state[1, 1] = np.array([0, 1, 0, 0])
    # place goal
    state[1, 2] = np.array([1, 0, 0, 0])

    a = findLoc(state, np.array([0, 0, 0, 1]))  # find grid position of player (agent)
    w = findLoc(state, np.array([0, 0, 1, 0]))  # find wall
    g = findLoc(state, np.array([1, 0, 0, 0]))  # find goal
    p = findLoc(state, np.array([0, 1, 0, 0]))  # find pit
    if (not a or not w or not g or not p):
        # print('Invalid grid. Rebuilding..')
        return initGridPlayer()

    return state.reshape(1, 64)


# Initialize grid so that goal, pit, wall, player are all randomly placed
def initGridRand():
    state = np.zeros((4, 4, 4))
    # place player
    state[randPair(0, 4)] = np.array([0, 0, 0, 1])
    # place wall
    state[randPair(0, 4)] = np.array([0, 0, 1, 0])
    # place pit
    state[randPair(0, 4)] = np.array([0, 1, 0, 0])
    # place goal
    state[randPair(0, 4)] = np.array([1, 0, 0, 0])

    a = findLoc(state, np.array([0, 0, 0, 1]))
    w = findLoc(state, np.array([0, 0, 1, 0]))
    g = findLoc(state, np.array([1, 0, 0, 0]))
    p = findLoc(state, np.array([0, 1, 0, 0]))
    # If any of the "objects" are superimposed, just call the function again to re-place
    if (not a or not w or not g or not p):
        # print('Invalid grid. Rebuilding..')
        return initGridRand()

    return state.reshape(1, 64)


def makeMove(state, action):
    state = state.reshape((4, 4, 4))

    # need to locate player in grid
    # need to determine what object (if any) is in the new grid spot the player is moving to
    player_loc = findLoc(state, np.array([0, 0, 0, 1]))
    wall = findLoc(state, np.array([0, 0, 1, 0]))
    goal = findLoc(state, np.array([1, 0, 0, 0]))
    pit = findLoc(state, np.array([0, 1, 0, 0]))
    state = np.zeros((4, 4, 4))

    if player_loc is None:
        return state.copy()

    # up (row - 1)
    if action == 0:
        new_loc = (player_loc[0] - 1, player_loc[1])
        if (new_loc != wall):
            if ((np.array(new_loc) <= (3, 3)).all() and (np.array(new_loc) >= (0, 0)).all()):
                state[new_loc][3] = 1
    # down (row + 1)
    elif action == 1:
        new_loc = (player_loc[0] + 1, player_loc[1])
        if (new_loc != wall):
            if ((np.array(new_loc) <= (3, 3)).all() and (np.array(new_loc) >= (0, 0)).all()):
                state[new_loc][3] = 1
    # left (column - 1)
    elif action == 2:
        new_loc = (player_loc[0], player_loc[1] - 1)
        if (new_loc != wall):
            if ((np.array(new_loc) <= (3, 3)).all() and (np.array(new_loc) >= (0, 0)).all()):
                state[new_loc][3] = 1
    # right (column + 1)
    elif action == 3:
        new_loc = (player_loc[0], player_loc[1] + 1)
        if (new_loc != wall):
            if ((np.array(new_loc) <= (3, 3)).all() and (np.array(new_loc) >= (0, 0)).all()):
                state[new_loc][3] = 1

    new_player_loc = findLoc(state, np.array([0, 0, 0, 1]))
    if (not new_player_loc):
        state[player_loc] = np.array([0, 0, 0, 1])
    # re-place pit
    state[pit][1] = 1
    # re-place wall
    state[wall][2] = 1
    # re-place goal
    state[goal][0] = 1

    return state.reshape(64, 1)


def getLoc(state, level):
    for i in range(0, 4):
        for j in range(0, 4):
            if (state[i, j][level] == 1):
                return i, j


def getReward(state):
    state = state.reshape(4, 4, 4)
    player_loc = getLoc(state, 3)
    pit = getLoc(state, 1)
    goal = getLoc(state, 0)
    if (player_loc == pit):
        return -10
    elif (player_loc == goal):
        return 10
    else:
        return -1


class Environment(object):
    def __init__(self):
        pass
        self.state = None

    def initialise(self):
        self.state = initGrid()

    def apply_action(self, action):
        self.state = makeMove(self.state, action)

    def get_reward(self):
        return getReward(self.state)

    def get_action_states(self):
        ''' Do not know in advancee'''
        action_states = np.ndarray((4, 64))
        for action in range(4):
            new_state = makeMove(self.state, action)
            action_states[action, :] = new_state.ravel()  # .reshape(1, 64)
        return action_states

    def get_action_rewards(self):
        ''' Do not know in advance'''
        action_rewards = np.empty(4)
        for action in range(4):
            action_state = makeMove(self.state, action)
            action_reward = getReward(action_state)
            action_rewards[action] = action_reward
        return action_rewards

    def display(self):
        display_state(self.state)

    @staticmethod
    def generate_experience():

        ssar = []
        env = Environment()
        env.initialise()
        old_state = env.state.reshape(64, 1)
        for _ in range(10):
            action_rewards = env.get_action_rewards()
            new_states = np.empty((64, 4))
            for action in range(4):
                new_state = makeMove(old_state, action)
                new_states[:, action] = new_state.ravel()
                reward = getReward(new_state)
                action_rewards[action] = reward

            ssar.append((old_state.reshape(64, 1).copy(), new_states, action_rewards))

            action = np.random.randint(0, 4)
            # print('Doing action %s' % action)
            old_state = makeMove(old_state, action)

        return ssar


def display_state(state):
    state = state.reshape(4, 4, 4)
    grid = np.zeros((4, 4), dtype='<U2')
    player_loc = findLoc(state, np.array([0, 0, 0, 1]))
    wall = findLoc(state, np.array([0, 0, 1, 0]))
    goal = findLoc(state, np.array([1, 0, 0, 0]))
    pit = findLoc(state, np.array([0, 1, 0, 0]))
    for i in range(0, 4):
        for j in range(0, 4):
            grid[i, j] = ' '

    if player_loc:
        grid[player_loc] = 'P'  # player
    if wall:
        grid[wall] = 'W'  # wall
    if goal:
        grid[goal] = '+'  # goal
    if pit:
        grid[pit] = '-'  # pit

    print(grid)


if __name__ == '__main__':
    grid = init_grid()
    print(grid.as_2d_array())

    # env = Environment()
    # env.initialise()
    # env.display()



    # print('Generating experience')
    # experience = []
    # for _ in range(100):
    #     exp = Environment.generate_experience()
    #     experience.extend(exp)
    #
    # N = len(experience)
    #
    # all_old_states = [exp[0] for exp in experience]
    # all_old_states = np.r_[all_old_states].reshape(N, 64)
    #
    # all_rewards = [exp[2] for exp in experience]
    # all_rewards = np.r_[all_rewards].reshape(N, 4)
    #
    # from rl.q_function import create_model
    # model = create_model()
    #
    # # Use the initial experience to fit the model
    # model.fit(all_old_states, all_rewards, epochs=100, batch_size=64, verbose=1)
    #
    # gamma = 0.9
    # print(model.predict(all_old_states[0, :].reshape(1, 64)))
    #
    # X = np.ndarray((N, 64))
    # y = np.ndarray((N, 4))
    # for i, (old_state, new_states, action_rewards) in enumerate(experience):
    #     print(old_state)
    #     new_state_q_values = model.predict(new_states.reshape(4, 64))
    #     new_state_max_q_values = new_state_q_values.max(axis=1)
    #     tgt = action_rewards + gamma * new_state_max_q_values
    #     X[i, :] = old_state.ravel()
    #     y[i, :] = tgt
    #
    # model.fit(X, y, epochs=100)
