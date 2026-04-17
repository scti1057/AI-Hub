class TicTacToeGame:
    def __init__(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_winner = None

    def make_move(self, row, col, player):
        """
        Attempts to place a marker on the board.
        Raises ValueError if the move is invalid.
        """
        if not (0 <= row < 3 and 0 <= col < 3):
            raise ValueError("Move is out of bounds.")
        
        if self.board[row][col] != ' ':
            raise ValueError("This cell is already occupied.")

        self.board[row][col] = player
        
        if self.check_winner(row, col, player):
            self.current_winner = player
            return True
        return False

    def check_winner(self, row, col, player):
        """
        Checks if the last move resulted in a win.
        """
        # Check row
        if all(self.board[row][i] == player for i in range(3)):
            return True
        # Check column
        if all(self.board[i][col] == player for i in range(3)):
            return True
        # Check diagonal 1
        if row == col and all(self.board[i][i] == player for i in range(3)):
            return True
        # Check diagonal 2
        if row + col == 2 and all(self.board[i][2 - i] == player for i in range(3)):
            return True
        
        return False

    def is_draw(self):
        """
        Checks if the board is full and no one has won.
        """
        return all(cell != ' ' for row in self.board for cell in row) and self.current_winner is None

    def reset(self):
        """
        Resets the game board.
        """
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_winner = None

    def get_board(self):
        """
        Returns the current state of the board.
        """
        return self.board