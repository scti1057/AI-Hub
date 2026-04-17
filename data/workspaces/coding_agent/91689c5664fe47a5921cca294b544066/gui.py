import tkinter as tk
from tkinter import messagebox

class TicTacToeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic Tac Toe AI")
        
        # Initialize the grid of buttons
        self.buttons = []
        self.create_grid()
        
        # Status label to show whose turn it is or who won
        self.status_label = tk.Label(self.root, text="Player X's Turn", font=('Arial', 14))
        self.status_label.pack(pady=10)

    def create_grid(self):
        self.grid_frame = tk.Frame(self.root)
        self.grid_frame.pack(padx=20, pady=20)
        
        for r in range(3):
            row_buttons = []
            for c in range(3):
                btn = tk.Button(
                    self.grid_frame, 
                    text="", 
                    font=('Arial', 20, 'bold'),
                    width=4, 
                    height=2,
                    command=lambda row=r, col=c: self.handle_click(row, col)
                )
                btn.grid(row=r, column=c, padx=5, pady=5)
                row_buttons.append(btn)
            self.buttons.append(row_buttons)

    def handle_click(self, row, col):
        # This will be integrated with game_logic.py in the next phase
        print(f"Button clicked at ({row}, {col})")
        # For now, just a visual indicator that the button works
        # self.buttons[row][col].config(text="X")

if __name__ == "__main__":
    root = tk.Tk()
    gui = TicTacToeGUI(root)
    root.mainloop()