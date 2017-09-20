import unittest

import numpy as np

from rl.examples.ant.state import AntState
from rl.examples.ant.value_function import AntActionValueFunction
from rl.core.state import IntExtState

class MyTestCase(unittest.TestCase):
    def test_vectorized_fit(self):
        states = States()
        target1 = np.random.randn(5, 2)
        target2 = np.random.randn(5, 2)
        targets = [target1, target2]

        value_function = AntActionValueFunction()

        original_loss = value_function.evaluate(states, targets)

        value_function.vectorized_fit(states, targets, epochs=100)

        fitted_loss = value_function.evaluate(states, targets)

        self.assertLess(fitted_loss[0], original_loss[0])
        self.assertLess(fitted_loss[1], original_loss[1])
        self.assertLess(fitted_loss[2], original_loss[2])

    def test___call__(self):
        state = AntState()
        int_ext_state = IntExtState(0, state)
        value_function = AntActionValueFunction()

        value = value_function(int_ext_state)

        self.assertIsInstance(value, np.ndarray)
        self.assertEqual(2, len(value))


class States(object):
    def __init__(self):
        self.values = np.random.randn(5, 11)

    def as_array(self):
        return self.values


if __name__ == '__main__':
    unittest.main()
