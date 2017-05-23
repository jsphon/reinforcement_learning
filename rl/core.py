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

    def generate_initial_training_data(self, num_epochs=10, max_epoch_len=100):
        # Refactor this to return state/action/rewards
        # Keep this for generating experience as we step through epochs
        state = self.model.get_new_state()

        experience = []

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

                experience.append((action_rewards, action_state_arr))

                next_action = np.random.randint(0, self.num_actions)
                state = action_states[next_action]

                if action_states[next_action].is_terminal:
                    print('Terminal')
                    break

        arr_action_rewards = np.r_[[itd[0] for itd in experience]]
        arr_action_states = np.concatenate([exp[1] for exp in experience])

        return arr_action_rewards, arr_action_states



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

