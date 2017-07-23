import numpy as np
import logging


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

    @property
    def value_function(self):
        return self.rl_system.value_function

    @property
    def policy(self):
        return self.rl_system.policy

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

    def generate_episodes(self, num_episodes, max_len=128):

        episodes = []
        for _ in range(num_episodes):
            episode = self.generate_episode(max_len=max_len)
            episodes.append(episode)
        return episodes

    def generate_episode(self, max_len=128):
        states = []
        actions = []
        rewards = []

        state = self.model.get_new_state()
        states.append(state)
        for _ in range(max_len):

            action = self.policy.choose_action(state)
            actions.append(action)
            new_state = self.model.apply_action(state, action)
            reward = self.reward_function(state, action, new_state)

            states.append(new_state)
            rewards.append(reward)

            if new_state.is_terminal:
                break
            else:
                state = new_state

        return Episode(states, actions, rewards)


class Episode(object):

    def __init__(self, states, actions, rewards):
        self.states = states
        self.actions = actions
        self.rewards = rewards

    def get_state_array(self):
        """

        :return: np.ndarray(num_states, state_size)
        """
        state_size = self.states[0].size
        num_states = len(self.states)
        dtype = self.states[0].array_dtype
        result = np.ndarray((num_states, state_size), dtype=dtype)
        for i in range(num_states):
            result[i, :] = self.states[i].as_array()
        return result

    def get_action_array(self):
        """
        :return np.ndarray(num_states)
        """
        return self.actions

    def get_reward_array(self):
        """

        :return np.ndarray(num_states):
        """
        return self.rewards

