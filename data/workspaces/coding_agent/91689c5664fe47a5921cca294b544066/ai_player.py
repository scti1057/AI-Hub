import random

class TicTacToeAI:
    """
    AI Player for Tic Tac Toe.
    Implements a basic strategic approach: Win > Block > Center > Random.
    """
    def __init__(self, marker):
        self.marker = marker
        self.opponent_marker = 'O' if marker == 'X' else 'X'

    def get_move(self, board):
        """
        Determines the best move for the AI.
        :param board: The current 3x3 game board (list of lists).
        :return: A tuple (row, col) representing the chosen move.
        """
        # 1. Try to win
        move = self._find_winning_move(board, self.marker)
        if move: return move

        # 2. Block opponent from winning
        move = self._find_winning_move(board, self.opponent_marker)
        if move: return move

        # 3. Take the center if available
        if board[1][1] == ' ':
            return (1, 1)

        # 4. Take a random available spot
        return self._get_random_move(board)

    def _find_winning_move(self, board, marker):
        """
        Checks if there is a move that results in a win for the given marker.
        """
        for r in range(3):
            for c in range(3):
                if board[r][c] == ' ':
                    # Simulate move
                    board[r][c] = marker
                    if self._check_win_condition(board, marker):
                        board[r][c] = ' ' # Reset
                        return (r, c)
                    board[r][c] = ' ' # Reset
        return None

    def _check_win_condition(self, board, marker):
        """
        Helper to check if the given marker has won the game.
        """
        # Rows and Cols
        for i in range(3):
            if all(board[i][j] == marker for j in range(3)): return True
            if all(board[j][i] == marker for j in range(3)): return True
        
        # Diagonals
        if all(board[i][i] == marker for i in range(3)): return True
        if all(board[i][2-i] == marker for i in range(3)): return True
        
        return False

    def _get_random_move(self, board):
        """
        Returns a random available move.
        """
        available_moves = []
        for r in range(3):
            for c in range(3):
                if board[r][c] == ' ':
                    available_moves.append((r, c))
        
        return random.choice(available_moves) if available_moves else None
