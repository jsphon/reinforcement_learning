import numpy as np

from rl.lib.timer import Timer


class RLTrainer(object):
    """
    For training Reinforcement Learning Systems
    """

    def __init__(self, rl_system):
        self.rl_system = rl_system

    def estimate_initial_value_function(self, episodes, epochs):
        states, action_rewards = self.rl_system.experience_generator.generate_initial_training_data(episodes)

        assert states.shape[1] == self.state_size
        assert action_rewards.shape[1] == self.num_actions

        self.value_function.fit(states, action_rewards, epochs=epochs)


class RLSystem(object):
    """
    From Sutton, a reinforcement system contains the following 4 components:
     - policy
     - reward function
     - value function
     - model
     """

    def __init__(self):
        self.policy = None
        self.reward_function = None
        self.action_value_function = None
        self.model = None
        self.get_new_state = None
        self.num_actions = 0
        self.state_size = 0
        self.gamma = 0.9
        self.state_class = State

        self.experience_generator = ExperienceGenerator(self)

    # def evaluate_policy(self):
    #
    #     value_function_class = type(self.action_value_function)
    #     result = value_function_class(self.policy)
    #
    #     states = self.state_class.all()
    #     return result

    def initialise_value_function(self, epochs=100):
        states, action_rewards = self.experience_generator.generate_initial_training_data(epochs)

        assert states.shape[1] == self.state_size
        assert action_rewards.shape[1] == self.num_actions

        self.value_function.fit(states, action_rewards, epochs=epochs)

        print('We have initialised the value function!')

        print('sampled action_rewards:')
        print(action_rewards[:5])

        print('trained action_rewards:')
        print(self.value_function.get_value(states[:5]))

    def train_model(self, num_epochs=10):
        for i in range(num_epochs):
            with Timer('Generating experience'):
                states, action_rewards, new_states, new_states_terminal = \
                    self.experience_generator.generate_experience(num_epochs=20)

            new_values = self.generate_action_target_values(new_states)
            new_values[new_states_terminal] = 0

            targets = action_rewards + self.gamma * new_values

            with Timer('Fitting model'):
                self.value_function.fit(states, targets, verbose=0)

    def generate_action_target_values(self, action_states):
        """
        Give an experience of action-states, calculate their values

        1st dimension of length N represents the number of experiences.

        2nd dimension represents an action.

        3rd dimension represents the state vector that the 2nd dimension's action takes us to

        So for the result, the 3rd dimension becomes the value of a subsequent action.

        i.e. result[0, 1, 2] would be the value from experience 0,
             of action 1 followed by action 2

        :param action_states: ndarray(N x num_actions x state_size)

        :return: ndarray(N x num_actions)
        """

        n = action_states.shape[0]
        # num_actions = action_states.shape[1]
        # state_size = action_states.shape[2]
        new_values0 = self.value_function.get_value(action_states.reshape(n * self.num_actions, self.state_size))
        new_values1 = new_values0.max(axis=1).reshape(n, self.num_actions)
        # new_values0 = self.value_function.get_value(action_states.reshape(n * num_actions, state_size))
        # new_values1 = new_values0.max(axis=1).reshape(n, num_actions)
        return new_values1


class ExperienceGenerator(object):
    def __init__(self, rl_system):

        self.rl_system = rl_system

    @property
    def model(self):
        return self.rl_system.model

    @property
    def num_actions(self):
        return self.rl_system.num_actions

    @property
    def state_size(self):
        return self.rl_system.state_size

    @property
    def reward_function(self):
        return self.rl_system.reward_function

    def generate_initial_training_data(self, num_epochs=20, max_epoch_len=100):
        '''
        Generate states / action-reward pairs, which can be used to initialise
        the value function to something more intelligent that randomness.

        :param num_epochs:
        :param max_epoch_len:
        :return: states, N x state_size ndarray
                 action-rewards N x num_actions ndarray
        '''

        state_history = []
        action_reward_history = []

        for _ in range(num_epochs):
            state = self.model.get_new_state()
            for _ in range(max_epoch_len):
                action_rewards = np.ndarray(self.num_actions)
                action_state_arr = np.ndarray((self.num_actions, self.state_size))
                action_states = []
                for action in range(self.num_actions):
                    new_state = self.model.apply_action(state, action)
                    action_rewards[action] = self.reward_function.get_reward(state, action, new_state)
                    action_states.append(new_state)
                    action_state_arr[action, :] = new_state.as_vector()

                state_history.append(state)
                action_reward_history.append(action_rewards)

                next_action = np.random.randint(0, self.num_actions)
                state = action_states[next_action]

                if state.is_terminal:
                    break

        arr_action_rewards = np.c_[action_reward_history]
        state_history_vectors = [sh.as_vector() for sh in state_history]
        arr_states = np.c_[state_history_vectors]

        return arr_states, arr_action_rewards

    def generate_experience(self, num_epochs=10, max_epoch_len=100):
        """
        Generate experience for training the model

        qval[action] = reward + gamma + maxQ

        Then we can fit the model with

        qval(old_state) ~= action_rewards + max(qval(new_action_states))

        :param num_epochs:
        :param max_epoch_len:
        :return:    old_state           N x state_size
                    action_rewards      N x num_actions
                    new_states          N x num_actions x state_size
                    new_states_terminal N x num_actions, True if new_state is terminal
        """

        state_history = []
        action_reward_history = []
        new_states_history = []
        new_states_terminal_history = []

        for _ in range(num_epochs):
            state = self.model.get_new_state()
            for _ in range(max_epoch_len):
                action_rewards = np.ndarray(self.num_actions)
                action_state_arr = np.ndarray((self.num_actions, self.state_size))
                action_states_terminal = np.ndarray(self.num_actions, dtype=np.bool)
                action_states = []
                for action in range(self.num_actions):
                    new_state = self.model.apply_action(state, action)
                    action_rewards[action] = self.reward_function.get_reward(state, action, new_state)
                    action_states.append(new_state)
                    action_state_arr[action, :] = new_state.as_vector()
                    action_states_terminal[action] = new_state.is_terminal

                state_history.append(state)
                action_reward_history.append(action_rewards)
                new_states_history.append(action_state_arr)
                new_states_terminal_history.append(action_states_terminal)

                next_action = np.random.randint(0, self.num_actions)
                state = action_states[next_action]

                if state.is_terminal:
                    break

        arr_action_rewards = np.c_[action_reward_history]
        state_history_vectors = [sh.as_vector() for sh in state_history]
        arr_states = np.c_[state_history_vectors]
        arr_new_states = np.r_[new_states_history]
        arr_new_states_terminal = np.r_[new_states_terminal_history]

        return arr_states, arr_action_rewards, arr_new_states, arr_new_states_terminal


class Model(object):
    def apply_action(self, state, action):
        ''' Might predict the next state and reward'''
        raise NotImplemented()


class State(object):
    def __init__(self):
        self.is_terminal = False
        self.size = 0
        self.array_dtype = np.float

    def as_array(self):
        """
        For input to the reward function fitting
        :return:
        """
        raise NotImplemented()

