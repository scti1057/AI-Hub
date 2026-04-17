import tkinter as tk
from tkinter import messagebox
from src.game_logic import Game
from src.constants import *


class TicTacToeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic Tac Toe")
        self.root.configure(bg=BG_COLOR)

        self.game = Game()
        self.current_player_label = tk.Label(root, text="Player X's Turn", font=FONT, bg=BG_COLOR, fg=TEXT_COLOR)
        self.current_player_label.pack(pady=10)

        self.board_frame = tk.Frame(root, bg=GRID_COLOR)
        self.board_frame.pack(pady=20)

        self.buttons = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                btn = tk.Button(
                    self.board_frame,
                    text=EMPTY,
                    font=FONT,
                    width=5,
                    height=2,
                    bg=BUTTON_BG,
                    command=lambda r=i, c=j: self.handle_click(r, c)
                )
                btn.grid(row=i, column=j, padx=5, pady=5)
                self.buttons[i][j] = btn

        self.reset_button = tk.Button(root, text="Reset Game", font=BUTTON_FONT, command=self.reset_game)
        self.reset_button.pack(pady=10)

    def handle_click(self, row: int, col: int):
        if self.game.make_move(row, col):
            self.buttons[row][col].config(text=self.game.current_player)
            if self.game.winner:
                messagebox.showinfo("Game Over", f"Player {self.game.winner} wins!")
                self.disable_buttons()
            elif self.game.is_draw:
                messagebox.showinfo("Game Over", "It's a draw!")
                self.disable_buttons()
            else:
                self.current_player_label.config(text=f"Player {self.game.current_player}'s Turn")

    def disable_buttons(self):
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                self.buttons[i][j].config(state="disabled")

    def reset_game(self):
        self.game.reset()
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                self.buttons[i][j].config(text=EMPTY, state="normal")
        self.current_player_label.config(text="Player X's Turn")


if __name__ == "__main__":
    root = tk.Tk()
    app = TicTacToeGUI(root)
    root.mainloop()
