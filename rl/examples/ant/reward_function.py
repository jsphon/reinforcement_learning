from rl.core.reward_function import RewardFunction

from rl.examples.ant.constants import FINDING_HOME, FINDING_FOOD, MOVE_LEFT, MOVE_RIGHT, HOME_POSITION, FOOD_POSITION


class AntRewardFunction(RewardFunction):

    def __call__(self, old_state, action, new_state):
        return self.get_reward(old_state, action, new_state)

    def get_reward(self, old_state, action, new_state):

        if new_state.internal_state==FINDING_FOOD:
            if new_state.external_state.position==FOOD_POSITION:
                return 10
        elif new_state.internal_state==FINDING_HOME:
            if new_state.external_state.position==HOME_POSITION:
                return 10

        return -1
