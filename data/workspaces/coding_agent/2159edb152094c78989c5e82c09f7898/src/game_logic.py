# Implementation of the Tic Tac Toe game engine
class TicTacToeLogic:
    def __init__(self):
        self.board = [None] * 9
        self.current_player = 'X'
        self.winner = None
        self.is_draw = False

    def make_move(self, position):
        if self.board[position] is None and not self.winner:
            self.board[position] = self.current_player
            if self.check_winner():
                self.winner = self.current_player
            elif None not in self.board:
                self.is_draw = True
            else:
                self.current_player = 'O' if self.current_player == 'X' else 'X'
            return True
        return False

    def check_winner(self):
        win_conditions = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8), # Rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8), # Cols
            (0, 4, 8), (2, 4, 6)             # Diagonals
        ]
        for a, b, c in win_conditions:
            if self.board[a] == self.board[b] == self.board[c] and self.board[a] is not None:
                return True
        return False

    def reset(self):
        self.__init__()