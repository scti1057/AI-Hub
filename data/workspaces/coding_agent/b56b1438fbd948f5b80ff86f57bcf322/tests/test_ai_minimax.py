import unittest
from src.game_logic import TicTacToe
from src.ai_agent import AIPlayer

class TestAIMinimax(unittest.TestCase):
    def setUp(self):
        self.game = TicTacToe()
        self.ai = AIPlayer('O', 'Hard')

    def test_minimax_win(self):
        # Set up a board where AI can win in one move
        self.game.board = [['X', 'X', ' '],
                           [' ', ' ', ' '],
                           [' ', ' ', ' ']]
        move = self.ai.get_move(self.game)
        self.assertEqual(move, (0, 2))  # Should block or win

    def test_minimax_block(self):
        # Set up a board where player can win in one move
        self.game.board = [[' ', ' ', ' '],
                           ['X', 'X', ' '],
                           [' ', ' ', ' ']]
        move = self.ai.get_move(self.game)
        self.assertEqual(move, (1, 2))  # Should block

    def test_minimax_center(self):
        # Test that AI picks center if available
        self.game.board = [[' ', ' ', ' '],
                           [' ', ' ', ' '],
                           [' ', ' ', ' ']]
        move = self.ai.get_move(self.game)
        self.assertEqual(move, (1, 1))  # Prefer center

if __name__ == '__main__':
    unittest.main()