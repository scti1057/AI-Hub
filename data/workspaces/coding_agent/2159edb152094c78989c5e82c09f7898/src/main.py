import tkinter as tk
from src.game_logic import TicTacToeLogic
from src.gui import TicTacToeGUI


def main():
    root = tk.Tk()
    logic = TicTacToeLogic()
    gui = TicTacToeGUI(root, logic)
    root.mainloop()


if __name__ == "__main__":
    main()