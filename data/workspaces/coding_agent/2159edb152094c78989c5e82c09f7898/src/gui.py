import tkinter as tk
from tkinter import messagebox

class TicTacToeGUI:
    def __init__(self, root, logic):
        self.root = root
        self.logic = logic
        self.root.title("Tic Tac Toe - Professional Edition")
        self.root.configure(bg="#2c3e50")
        
        self.buttons = []
        self.setup_ui()

    def setup_ui(self):
        self.main_frame = tk.Frame(self.root, bg="#2c3e50")
        self.main_frame.pack(padx=20, pady=20)

        self.status_label = tk.Label(
            self.main_frame, text=f"Player {self.logic.current_player}'s Turn",
            font=("Helvetica", 16, "bold"), fg="#ecf0f1", bg="#2c3e50"
        )
        self.status_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        for i in range(9):
            btn = tk.Button(
                self.main_frame, text="", font=("Helvetica", 20, "bold"),
                width=4, height=2, bg="#34495e", fg="#ecf0f1",
                command=lambda i=i: self.on_click(i)
            )
            btn.grid(row=(i // 3) + 1, column=i % 3, padx=5, pady=5)
            self.buttons.append(btn)

        self.reset_btn = tk.Button(
            self.main_frame, text="Reset Game", font=("Helvetica", 12),
            command=self.reset_game, bg="#e74c3c", fg="white"
        )
        self.reset_btn.grid(row=4, column=0, columnspan=3, pady=20)

    def on_click(self, index):
        if self.logic.make_move(index):
            self.buttons[index].config(text=self.logic.board[index], state="disabled", disabledforeground="#bdc3c7")
            self.update_status()

    def update_status(self):
        if self.logic.winner:
            self.status_label.config(text=f"Player {self.logic.winner} Wins!")
            messagebox.showinfo("Game Over", f"Player {self.logic.winner} has won the game!")
        elif self.logic.is_draw:
            self.status_label.config(text="It's a Draw!")
            messagebox.showinfo("Game Over", "The game ended in a draw!")
        else:
            self.status_label.config(text=f"Player {self.logic.current_player}'s Turn")

    def reset_game(self):
        self.logic.reset()
        for btn in self.buttons:
            btn.config(text="", state="normal")
        self.status_label.config(text=f"Player {self.logic.current_player}'s Turn")