import tensorflow as tf


class ExperienceGenerator(object):

    def __init__(self, rl_system):
        self.rl_system = rl_system

    def generate(self, initial_state):

        """
        while not rl_system.model.is_terminal(s):
            choose action
            apply acction

        Args:
            initial_state:

        Returns:

        """