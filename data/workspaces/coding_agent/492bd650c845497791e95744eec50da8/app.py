from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

# Game State
state = {
    "board": ["" for _ in range(9)],
    "turn": "X",
    "winner": None,
    "is_draw": False
}

WIN_COMBINATIONS = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8), # Rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8), # Cols
    (0, 4, 8), (2, 4, 6)             # Diagonals
]

def check_winner():
    for a, b, c in WIN_COMBINATIONS:
        if state["board"][a] == state["board"][b] == state["board"][c] != "":
            return state["board"][a]
    if "" not in state["board"]:
        return "Draw"
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/move', methods=['POST'])
def move():
    data = request.json
    index = data.get('index')
    
    if index is None or index < 0 or index > 8:
        return jsonify({"error": "Invalid index"}), 400
    
    if state["board"][index] != "" or state["winner"]:
        return jsonify({"error": "Invalid move"}), 400
    
    # Update board
    state["board"][index] = state["turn"]
    
    # Check for winner
    winner = check_winner()
    if winner:
        state["winner"] = winner
        if winner == "Draw":
            state["is_draw"] = True
    else:
        # Switch turn
        state["turn"] = "O" if state["turn"] == "X" else "X"
    
    return jsonify(state)

@app.route('/reset', methods=['POST'])
def reset():
    state["board"] = ["" for _ in range(9)]
    state["turn"] = "X"
    state["winner"] = None
    state["is_draw"] = False
    return jsonify(state)

if __name__ == '__main__':
    app.run(debug=True)