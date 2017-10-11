import numpy as np
from rl.core.state import StateList


class ExperienceGenerator(object):
    def __init__(self, rl_system):

        self.rl_system = rl_system

    @property
    def model(self):
        return self.rl_system.model

    @property
    def reward_function(self):
        return self.rl_system.reward_function

    @property
    def policy(self):
        return self.rl_system.policy

    def generate_episodes(self, num_episodes, max_len=128):

        episodes = EpisodeList()
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

            action = self.rl_system.choose_action(state)
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


class Experience(object):
    def get_training_states(self):
        """

        Returns: StatesList

        """
        raise NotImplemented()

    def get_training_states(self):
        raise NotImplemented()

    def get_training_actions(self):
        raise NotImplemented()

    def get_training_rewards(self):
        raise NotImplemented()


# class StatesList(list):
#
#     def get_training_length(self):
#         return len(self)
#
#     def as_array(self):
#         state_size = self[0].size
#         num_training_states = len(self)
#         dtype = self[0].array_dtype
#         result = np.ndarray((num_training_states, state_size), dtype=dtype)
#         for i in range(num_training_states):
#             result[i, :] = self[i].as_array()
#         return result


class Episode(Experience):
    def __init__(self, states, actions, rewards):
        self.states = states
        self.actions = actions
        self.rewards = rewards

    def get_training_length(self):
        return len(self.states) - 1

    def get_training_states(self):
        return StateList(self.states[:-1])

    def get_training_actions(self):
        return self.actions

    def get_training_rewards(self):
        return self.rewards


class EpisodeList(Experience):
    def __init__(self, episodes=None):
        self.episodes = episodes or []

    def __len__(self):
        return len(self.episodes)

    def __iter__(self):
        return self.episodes.__iter__()

    def append(self, episode):
        self.episodes.append(episode)

    def get_training_length(self):
        return sum(episode.get_training_length() for episode in self.episodes)

    def get_training_actions(self):
        actions_lst = []
        for episode in self.episodes:
            actions_lst.append(episode.get_training_actions())
        return np.concatenate(actions_lst, axis=0)

    def get_training_rewards(self):
        rewards_lst = []
        for episode in self.episodes:
            rewards_lst.append(episode.get_training_rewards())
        return np.concatenate(rewards_lst, axis=0)

    def get_training_states(self):
        training_states = StateList()
        for episode in self.episodes:
            training_states.extend(episode.get_training_states())
        return training_states

    def get_training_states_array(self):
        state_arrays_lst = []

        for episode in self.episodes:
            state_arrays_lst.append(episode.get_training_states_array())
        return np.concatenate(state_arrays_lst, axis=0)
