import numba
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

    def generate_initial_training_data(self, num_epochs=10, max_epoch_len=100):
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

                if action_states[next_action].is_terminal:
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
                    new_states   N x num_actions x state_size
        '''
        state = self.model.get_new_state()

        state_history = []
        action_reward_history = []
        new_states_history = []

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
                new_states_history.append(action_state_arr)

                next_action = np.random.randint(0, self.num_actions)
                state = action_states[next_action]

                if action_states[next_action].is_terminal:
                    break

        arr_action_rewards = np.c_[action_reward_history]
        state_history_vectors = [sh.as_vector() for sh in state_history]
        arr_states = np.c_[state_history_vectors]
        arr_new_states = np.r_[new_states_history]

        return arr_states, arr_action_rewards, arr_new_states


class Policy(object):
    def get_action(self, action_values):
        raise NotImplemented()


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
