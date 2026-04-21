async function makeMove(index) {
    const response = await fetch('/move', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ index: index })
    });
    const data = await response.json();
    if (response.ok) {
        updateBoard(data);
    } else {
        alert(data.error || 'Move failed');
    }
}

async function resetGame() {
    const response = await fetch('/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    });
    const data = await response.json();
    updateBoard(data);
}

function updateBoard(state) {
    const cells = document.querySelectorAll('.cell');
    state.board.forEach((val, i) => {
        cells[i].textContent = val;
        cells[i].classList.remove('x', 'o');
        if (val) cells[i].classList.add(val.toLowerCase());
    });

    const status = document.getElementById('status');
    if (state.winner) {
        status.textContent = state.is_draw ? "It's a Draw!" : `Player ${state.winner} Wins!`;
    } else {
        status.textContent = `Player ${state.turn}'s Turn`;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const cells = document.querySelectorAll('.cell');
    cells.forEach((cell, index) => {
        cell.addEventListener('click', () => makeMove(index));
    });

    document.getElementById('reset-btn').addEventListener('click', resetGame);
    
    // Initial state
    resetGame();
});