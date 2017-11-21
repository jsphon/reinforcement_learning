import tensorflow as tf
from rl.tf.environments.line_world.constants import TARGET


class RewardFunction(object):

    def action_reward(self, state, action, next_state):
        """
        Receives state, action, state as input, hence the name
        Args:
            state:
            action:
            next_state:

        Returns:

        """

        to_home = tf.equal(next_state, TARGET)

        return tf.case({
            to_home: lambda: 10,
            },
            default = lambda: -1,
            exclusive=True)

    def state_rewards(self, state, next_states):

        r0 = self.action_reward(state, 0, next_states[0])
        r1 = self.action_reward(state, 1, next_states[1])
        return tf.stack([r0, r1])
