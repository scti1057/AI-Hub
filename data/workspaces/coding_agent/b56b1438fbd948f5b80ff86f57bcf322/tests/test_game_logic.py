import unittest
from src.game_logic import TicTacToe

class TestGameLogic(unittest.TestCase):
    def setUp(self):
        self.game = TicTacToe()

    def test_initial_board_state(self):
        self.assertEqual(self.game.board, [[' ' for _ in range(3)] for _ in range(3)])
        self.assertEqual(self.game.current_player, 'X')

    def test_make_move_valid(self):
        self.assertTrue(self.game.make_move(0, 0))
        self.assertEqual(self.game.board[0][0], 'X')

    def test_make_move_invalid(self):
        self.game.make_move(0, 0)
        self.assertFalse(self.game.make_move(0, 0))  # Already taken

    def test_check_winner_row(self):
        self.game.make_move(0, 0)
        self.game.make_move(1, 0)
        self.game.make_move(0, 1)
        self.game.make_move(1, 1)
        self.game.make_move(0, 2)
        self.assertTrue(self.game.check_winner())
        self.assertEqual(self.game.winner, 'X')

    def test_check_winner_column(self):
        self.game.make_move(0, 0)
        self.game.make_move(0, 1)
        self.game.make_move(1, 0)
        self.game.make_move(1, 1)
        self.game.make_move(2, 0)
        self.assertTrue(self.game.check_winner())
        self.assertEqual(self.game.winner, 'X')

    def test_check_winner_diagonal(self):
        self.game.make_move(0, 0)
        self.game.make_move(0, 1)
        self.game.make_move(1, 1)
        self.game.make_move(0, 2)
        self.game.make_move(2, 2)
        self.assertTrue(self.game.check_winner())
        self.assertEqual(self.game.winner, 'X')

    def test_check_draw(self):
        moves = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]
        for i, (row, col) in enumerate(moves):
            if i % 2 == 0:
                self.game.make_move(row, col)
            else:
                self.game.make_move(row, col)
        self.assertTrue(self.game.is_board_full())
        self.assertFalse(self.game.check_winner())

if __name__ == '__main__':
    unittest.main()