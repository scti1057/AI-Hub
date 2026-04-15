from src.tictactoe.core import TicTacToeGame

def test_win_condition():
    game = TicTacToeGame()
    game.make_move(0, 0) # X
    game.make_move(0, 1) # O
    game.make_move(0, 2) # X
    game.make_move(1, 1) # O
    game.make_move(1, 0) # X
    game.make_move(1, 2) # O
    game.make_move(2, 0) # X - Wins
    assert game.winner == 'X'
    assert game.game_over == True

def test_draw_condition():
    game = TicTacToeGame()
    moves = [(0,0), (0,1), (0,2), (1,1), (1,0), (1,2), (2,1), (2,0), (2,2)]
    for r, c in moves:
        game.make_move(r, c)
    assert game.winner is None
    assert game.game_over == True

if __name__ == '__main__':
    test_win_condition()
    test_draw_condition()
    print('All core tests passed!')