from typing import List, Optional

class Game:
    def __init__(self):
        self.board = [[None for _ in range(3)] for _ in range(3)]
        self.current_player = "X"
        self.winner = None
        self.is_draw = False

    def make_move(self, row: int, col: int) -> bool:
        if self.board[row][col] is not None:
            return False
        self.board[row][col] = self.current_player
        self.check_game_end()
        if not self.winner and not self.is_draw:
            self.current_player = "O" if self.current_player == "X" else "X"
        return True

    def check_game_end(self):
        # Check rows
        for row in self.board:
            if row[0] == row[1] == row[2] and row[0] is not None:
                self.winner = row[0]
                return

        # Check columns
        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] and self.board[0][col] is not None:
                self.winner = self.board[0][col]
                return

        # Check diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] and self.board[0][0] is not None:
            self.winner = self.board[0][0]
            return
        if self.board[0][2] == self.board[1][1] == self.board[2][0] and self.board[0][2] is not None:
            self.winner = self.board[0][2]
            return

        # Check for draw
        if all(cell is not None for row in self.board for cell in row):
            self.is_draw = True

    def reset(self):
        self.__init__()
