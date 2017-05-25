import unittest
import numpy as np
from rl.environment import rand_pair
import rl.environment as rle
from rl import grid_system


N = 1000


class MyTestCase(unittest.TestCase):

    def test_single(self):
        gvf = grid_system.GridValueFunction()

        state = np.zeros((1, 64))
        values = gvf.get_value(state)

        self.assertEqual((1, 4), values.shape)
        print(values)

    def test_multi(self):
        gvf = grid_system.GridValueFunction()

        state = np.zeros((2, 64))
        values = gvf.get_value(state)

        print(values)

    def test_generate_action_target_values(self):

        grid_sys = grid_system.GridSystem()
        grid_sys.initialise_value_function(epochs=10)

        states, action_rewards, new_states = grid_sys.generate_experience(num_epochs=10)

        new_state_values = grid_sys.generate_action_target_values(new_states)

        N = len(new_state_values)
        expected = np.ndarray((N, grid_sys.num_actions))
        for i in range(N):
            for action in range(grid_sys.num_actions):
                action_values = grid_sys.value_function.get_value(new_states[i][action].reshape(1, grid_sys.state_size))
                expected[i, action] = action_values.max()

        np.testing.assert_almost_equal(expected, new_state_values)

    #
    # def test_init_grid(self):
    #     grid = rle.init_grid()
    #     self.assertEqual((0, 1), grid.player)
    #     self.assertEqual((2, 2), grid.wall)
    #     self.assertEqual((3, 3), grid.goal)
    #     self.assertEqual((3, 3), grid.goal)
    #
    # def test_as_one_hot(self):
    #     grid = rle.init_grid()
    #     vec = grid.as_one_hot()
    #
    #     self.assertIsInstance(vec, np.ndarray)
    #     self.assertEqual(64, vec.shape[0])
    #     self.assertTrue(vec[1]) # Player
    #     self.assertTrue(vec[26]) # Wall
    #     self.assertTrue(vec[37]) # Pit
    #     self.assertTrue(vec[63]) # Goal
    #
    # def test_as_2d_array(self):
    #
    #     grid = rle.init_grid()
    #
    #     actual = grid.as_2d_array()
    #
    #     expected = np.empty((4, 4), dtype='<U2')
    #     expected[:] = ' '
    #     expected[0, 1] = 'P'
    #     expected[2, 2] = 'W'
    #     expected[1, 1] = '-'
    #     expected[3, 3] = '+'
    #
    #     np.testing.assert_array_equal(expected, actual)
    #
    # def test_move_up_into_edge(self):
    #     grid = rle.init_grid()
    #     grid.move_up()
    #     self.assertEqual((0, 1), grid.player)
    #
    # def test_move_up_into_wall(self):
    #     grid = rle.init_grid()
    #     grid.player = (3, 2)
    #     grid.move_up()
    #     self.assertEqual((3, 2), grid.player)
    #
    # def test_move_up_into_space(self):
    #     grid = rle.init_grid()
    #     grid.player = (1, 2)
    #     grid.move_up()
    #     self.assertEqual((0, 2), grid.player)
    #
    # def test_move_down(self):
    #     grid = rle.init_grid()
    #     grid.move_down()
    #     self.assertEqual((1, 1), grid.player)
    #
    # def test_move_down_into_edge(self):
    #     grid = rle.init_grid()
    #     grid.player = (3, 0)
    #     grid.move_down()
    #     self.assertEqual((3, 0), grid.player)
    #
    # def test_move_down_into_wall(self):
    #     grid = rle.init_grid()
    #     grid.player = (1, 2)
    #     grid.move_down()
    #     self.assertEqual((1, 2), grid.player)
    #
    # def test_move_left(self):
    #     grid = rle.init_grid()
    #     grid.move_left()
    #     self.assertEqual((0, 0), grid.player)
    #
    # def test_move_left_into_edge(self):
    #     grid = rle.init_grid()
    #     grid.player = (0, 0)
    #     grid.move_left()
    #     self.assertEqual((0, 0), grid.player)
    #
    # def test_move_left_into_wall(self):
    #     grid = rle.init_grid()
    #     grid.player = (2, 3)
    #     grid.move_left()
    #     self.assertEqual((2, 3), grid.player)
    #
    # def test_move_right(self):
    #     grid = rle.init_grid()
    #     grid.move_right()
    #     self.assertEqual((0, 2), grid.player)
    #
    # def test_move_right_into_edge(self):
    #     grid = rle.init_grid()
    #     grid.player = (0, 3)
    #     grid.move_right()
    #     self.assertEqual((0, 3), grid.player)
    #
    # def test_move_right_into_wall(self):
    #     grid = rle.init_grid()
    #     grid.player = (2, 1)
    #     grid.move_right()
    #     self.assertEqual((2, 1), grid.player)
    #
    # def test_get_reward_pit(self):
    #     grid = rle.init_grid()
    #     grid.player = grid.pit
    #     self.assertEqual(-10, grid.get_reward())
    #
    # def test_get_reward_goal(self):
    #     grid = rle.init_grid()
    #     grid.player = grid.goal
    #     self.assertEqual(10, grid.get_reward())
    #
    # def test_get_reward_other(self):
    #     grid = rle.init_grid()
    #     self.assertEqual(-1, grid.get_reward())
    #
    # def test_rand_pair(self):
    #
    #     for _ in range(N):
    #         rp = rand_pair(0, 4)
    #         self.assertEqual(2, len(rp))
    #         self.assertIn(rp[0], (0, 1, 2, 3))
    #         self.assertIn(rp[1], (0, 1, 2, 3))



if __name__ == '__main__':
    unittest.main()
