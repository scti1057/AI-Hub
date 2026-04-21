# Minimax AI implementation for Tic Tac Toe

import random

class TicTacToeBot:
    def __init__(self, symbol):
        self.symbol = symbol
        self.opponent_symbol = 'O' if symbol == 'X' else 'X'

    def get_best_move(self, board):
        # Minimax algorithm to find the optimal move
        best_score = -float('inf')
        move = None
        
        available_moves = [i for i, spot in enumerate(board) if spot == ' ']
        random.shuffle(available_moves)

        for m in available_moves:
            board[m] = self.symbol
            score = self._minimax(board, 0, False)
            board[m] = ' '
            if score > best_score:
                best_score = score
                move = m
        return move

    def _minimax(self, board, depth, is_maximizing):
        # Check for terminal states
        result = self._check_winner(board)
        if result == self.symbol: return 10 - depth
        if result == self.opponent_symbol: return depth - 10
        if ' ' not in board: return 0

        if is_maximizing:
            best_score = -float('inf')
            for i in range(9):
                if board[i] == ' ':
                    board[i] = self.symbol
                    score = self._minimax(board, depth + 1, False)
                    board[i] = ' '
                    best_score = max(score, best_score)
            return best_score
        else:
            best_score = float('inf')
            for i in range(9):
                if board[i] == ' ':
                    board[i] = self.opponent_symbol
                    score = self._minimax(board, depth + 1, True)
                    board[i] = ' '
                    best_score = min(score, best_score)
            return best_score

    def _check_winner(self, board):
        win_conditions = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8], # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8], # Cols
            [0, 4, 8], [2, 4, 6]             # Diagonals
        ]
        for condition in win_conditions:
            if board[condition[0]] == board[condition[1]] == board[condition[2]] != ' ':
                return board[condition[0]]
        return None