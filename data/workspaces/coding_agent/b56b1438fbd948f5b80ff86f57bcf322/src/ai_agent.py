import random

class AIPlayer:
    def __init__(self, difficulty="easy"):
        self.difficulty = difficulty

    def get_move(self, board):
        if self.difficulty == "easy":
            return self._get_easy_move(board)
        elif self.difficulty == "hard":
            return self._get_hard_move(board)
        else:
            raise ValueError("Invalid difficulty level. Use 'easy' or 'hard'.")

    def _get_easy_move(self, board):
        # Simple random move
        empty_cells = [(i, j) for i in range(3) for j in range(3) if board[i][j] is None]
        return random.choice(empty_cells) if empty_cells else None

    def _get_hard_move(self, board):
        # Minimax algorithm for unbeatable AI
        best_score = float('-inf')
        best_move = None
        
        for i in range(3):
            for j in range(3):
                if board[i][j] is None:
                    board[i][j] = 'O'  # AI is always 'O'
                    score = self._minimax(board, 0, False)
                    board[i][j] = None
                    if score > best_score:
                        best_score = score
                        best_move = (i, j)
        
        return best_move

    def _minimax(self, board, depth, is_maximizing):
        # Check for terminal states
        winner = self._check_winner(board)
        if winner == 'O':
            return 10 - depth
        elif winner == 'X':
            return depth - 10
        elif self._is_board_full(board):
            return 0

        if is_maximizing:
            best_score = float('-inf')
            for i in range(3):
                for j in range(3):
                    if board[i][j] is None:
                        board[i][j] = 'O'
                        score = self._minimax(board, depth + 1, False)
                        board[i][j] = None
                        best_score = max(score, best_score)
            return best_score
        else:
            best_score = float('inf')
            for i in range(3):
                for j in range(3):
                    if board[i][j] is None:
                        board[i][j] = 'X'
                        score = self._minimax(board, depth + 1, True)
                        board[i][j] = None
                        best_score = min(score, best_score)
            return best_score

    def _check_winner(self, board):
        # Check rows
        for row in board:
            if row[0] == row[1] == row[2] and row[0] is not None:
                return row[0]
        
        # Check columns
        for col in range(3):
            if board[0][col] == board[1][col] == board[2][col] and board[0][col] is not None:
                return board[0][col]
        
        # Check diagonals
        if board[0][0] == board[1][1] == board[2][2] and board[0][0] is not None:
            return board[0][0]
        if board[0][2] == board[1][1] == board[2][0] and board[0][2] is not None:
            return board[0][2]
        
        return None

    def _is_board_full(self, board):
        return all(cell is not None for row in board for cell in row)
