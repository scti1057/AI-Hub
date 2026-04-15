class TicTacToeGame:
    def __init__(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'
        self.winner = None
        self.game_over = False

    def make_move(self, row, col):
        if self.game_over or not (0 <= row < 3 and 0 <= col < 3) or self.board[row][col] != ' ':
            return False

        self.board[row][col] = self.current_player
        if self._check_winner(row, col):
            self.winner = self.current_player
            self.game_over = True
        elif self._is_draw():
            self.game_over = True
        else:
            self.current_player = 'O' if self.current_player == 'X' else 'X'
        return True

    def _check_winner(self, row, col):
        # Check row
        if all(self.board[row][i] == self.current_player for i in range(3)):
            return True
        # Check col
        if all(self.board[i][col] == self.current_player for i in range(3)):
            return True
        # Check diagonals
        if row == col and all(self.board[i][i] == self.current_player for i in range(3)):
            return True
        if row + col == 2 and all(self.board[i][2-i] == self.current_player for i in range(3)):
            return True
        return False

    def _is_draw(self):
        return all(cell != ' ' for row in self.board for cell in row)

    def get_board(self):
        return self.board