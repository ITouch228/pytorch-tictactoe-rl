"""Абсолютная доска (X=+1, O=-1) и генератор greedy-партий для показа.

Игровая логика (Board) хранит перспективу «чей ход», а для отрисовки нужна
абсолютная раскраска X/O - здесь единая реализация этого перехода.
"""

import numpy as np

from agent.agent import Agent
from game.game import Game


class AbsoluteBoard:
    """Доска в абсолютной перспективе для отрисовки: X=+1, O=-1, пусто=0."""

    def __init__(self) -> None:
        self.cells = np.zeros(9, dtype=np.float32)
        self.player = 1.0

    def place(self, a: int) -> None:
        """Ставит фишку текущего игрока в клетку a и передаёт ход."""
        self.cells[a] = self.player
        self.player = -self.player

    def reset(self) -> None:
        self.cells = np.zeros(9, dtype=np.float32)
        self.player = 1.0


def play_greedy_game(agent: Agent, max_moves: int = 9) -> list[np.ndarray]:
    """Greedy-партия для показа. Возвращает список досок в абсолютной
    перспективе (+1 = X, -1 = O), начиная с пустой.
    """
    game = Game()
    board = AbsoluteBoard()
    boards = [board.cells.copy()]

    while not game.is_finished() and len(boards) <= max_moves:
        a, _ = agent.select_action(game.state(), game.legal_moves(), greedy=True)
        game.step(a)
        board.place(a)
        boards.append(board.cells.copy())

    return boards
