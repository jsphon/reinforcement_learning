import random

import numpy as np


class RLSystem(object):
    ''' From Sutton, a reinforcement system contains the following 4 components:
     - policy
     - reward function
     - value function
     - model
     '''

    def __init__(self):

        self.policy = None
        self.reward_function = None
        self.value_function = None
        self.model = None
        self.get_new_state = None
        self.num_actions = 0
        self.state_size = 0

    def initialise_value_function(self, epochs=100):

        states, action_rewards = self.generate_initial_training_data()

        assert states.shape[1] == self.state_size
        assert action_rewards.shape[1] == self.num_actions

        self.value_function.fit(states, action_rewards, epochs=epochs)

        print('We have initialised the value function!')

        print('sampled action_rewards:')
        print(action_rewards[:5])

        print('trained action_rewards:')
        print(self.value_function.get_value(states[:5]))

    def generate_initial_training_data(self, num_epochs=20, max_epoch_len=100):
        '''
        Generate states / action-reward pairs, which can be used to initialise
        the value function to something more intelligent that randomness.

        :param num_epochs:
        :param max_epoch_len:
        :return: states, N x state_size ndarray
                 action-rewards N x num_actions ndarray
        '''
        state = self.model.get_new_state()

        state_history = []
        action_reward_history = []

        for _ in range(num_epochs):
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
                    state = self.model.get_new_state()
                    break

        arr_action_rewards = np.c_[action_reward_history]
        state_history_vectors = [sh.as_vector() for sh in state_history]
        arr_states = np.c_[state_history_vectors]

        return arr_states, arr_action_rewards

    def generate_experience(self, num_epochs=10, max_epoch_len=100):
        '''
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
        '''
        state = self.model.get_new_state()

        state_history = []
        action_reward_history = []
        new_states_history = []
        new_states_terminal_history = []

        for _ in range(num_epochs):
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

                if action_states[next_action].is_terminal:
                    break

        arr_action_rewards = np.c_[action_reward_history]
        state_history_vectors = [sh.as_vector() for sh in state_history]
        arr_states = np.c_[state_history_vectors]
        arr_new_states = np.r_[new_states_history]
        arr_new_states_terminal = np.r_[new_states_terminal_history]

        return arr_states, arr_action_rewards, arr_new_states, arr_new_states_terminal

    def generate_action_target_values(self, action_states):
        '''
        Give an experience of action-states, calculate their values

        1st dimension of length N represents the number of experiences.

        2nd dimension represents an action.

        3rd dimension represents the state vector that the 2nd dimension's action takes us to

        So for the result, the 3rd dimension becomes the value of a subsequent action.

        i.e. result[0, 1, 2] would be the value from experience 0,
             of action 1 followed by action 2

        :param action_states: ndarray(N x num_actions x state_size)

        :return: ndarray(N x num_actions)
        '''

        N = action_states.shape[0]
        new_values0 = self.value_function.get_value(action_states.reshape(N * self.num_actions, self.state_size))
        new_values1 = new_values0.max(axis=1).reshape(N, self.num_actions)
        return new_values1


class Policy(object):
    def choose_action(self, action_values):
        raise NotImplemented()


class EpsilonGreedyPolicy(Policy):
    def __init__(self):
        self.epsilon = 0.1

    def choose_action(self, action_values, num_actions):
        if random.random() < self.epsilon:  # choose random action
            action = np.random.randint(0, num_actions)
        else:  # choose best action from Q(s,a) values
            action = (np.argmax(action_values))
        print('Choosing action %s from %s' % (action, str(action_values)))
        return action


class RewardFunction(object):
    def get_reward(self, old_state, action, new_state):
        raise NotImplemented()


class ValueFunction(object):
    def get_value(self, state):
        raise NotImplemented()


class Model(object):
    def apply_action(self, state, action):
        ''' Might predict the next state and reward'''
        raise NotImplemented()


class State(object):
    def __init__(self):
        self.is_terminal = False


class Episode(object):
    """
    A reinforcement learning episde
    """

    def __init__(self, rl_system, epsilon=0.0):
        self.rl_system = rl_system
        self.current_state = None
        self.epsilon = epsilon

        self.state_history = None
        self.reward_history = None
        self.action_history = None

        self.total_reward = None

    def initialise(self):
        self.current_state = self.rl_system.model.get_new_state()

    def choose_action(self, state):
        action_values = self.rl_system.value_function.get_value(state.as_vector().reshape(1, -1))
        chosen_action = self.rl_system.policy.choose_action(action_values, self.rl_system.num_actions)
        return chosen_action

    def take_action(self, action):
        print('Taking action %s' % action)
        next_state = self.rl_system.model.apply_action(self.current_state, action)
        self.current_state = next_state

    def play(self, max_epochs=10):
        current_state = self.rl_system.model.get_new_state()

        action_history = list()
        action_history.append(None)
        state_history = list()
        state_history.append(current_state.as_vector())

        reward_history = [0]

        total_reward = 0
        count = 0
        while not current_state.is_terminal and count < max_epochs:
            action = self.choose_action(current_state)
            next_state = self.rl_system.model.apply_action(current_state, action)
            reward = self.rl_system.reward_function.get_reward(current_state, action, next_state)
            total_reward += reward

            action_history.append(action)
            state_history.append(next_state)
            reward_history.append(reward)

            current_state = next_state
            count += 1

        self.action_history = action_history
        self.state_history = state_history
        self.reward_history = reward_history
        self.total_reward = total_reward
        print('Total reward %s' % total_reward)
