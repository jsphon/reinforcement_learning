import logging

logging.getLogger("tensorflow").setLevel(logging.WARNING)

import unittest
import numpy as np
import tensorflow as tf
from rl.lib.timer import Timer
from rl.tests.tf.utils import evaluate_tensor

from rl.tf.core.value_function import ValueFunctionBuilder, squared_loss
from rl.tf.environments.line_world.system import LineWorldSystem
from rl.tf.core.training.action_target_calculator import QLearningActionTargetCalculator
from rl.tf.core.training.target_array_calculator import ModelBasedTargetArrayCalculator
from rl.tf.core.training.trainer import Trainer

TRAIN_STEPS = 1000
LEARNING_RATE = 0.001


class TrainerTests(tf.test.TestCase):

    def test_1(self):

        a_states = np.arange(10)
        t_states = tf.constant(a_states, dtype=tf.int32)

        lws = LineWorldSystem()

        action_target_calculator = QLearningActionTargetCalculator(lws)
        target_array_calculator = ModelBasedTargetArrayCalculator(lws, action_target_calculator)

        t_action_values = lws.action_value_function.vectorized(t_states)

        trainer = Trainer(
            t_states,
            target_array_calculator,
            lws.action_value_function
        )

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            print(trainer.y_snapshot.eval())

            trainer.train_once(sess)
            sess.run(trainer.train_op)

            print(trainer.y_snapshot.eval())


if __name__ == '__main__':
    unittest.main()
