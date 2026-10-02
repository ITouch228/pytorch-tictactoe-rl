import numpy as np

from game.board import Board


class Game:
    """Управляет одной партией: ходы, итог."""

    def __init__(self) -> None:
        self.board: Board = Board()
        self.move_count: int = 0
        self.finished: bool = False
        self._outcome: int | None = None

    def state(self) -> np.ndarray:
        return self.board.to_array()

    def legal_moves(self) -> list[int]:
        return self.board.legal_moves()

    def is_finished(self) -> bool:
        return self.finished

    def is_agent_turn(self, agent_as_x: bool) -> bool:
        """Ходит агент, играющий сторону agent_as_x?"""
        return (self.move_count % 2 == 0) == agent_as_x

    def step(self, a: int) -> None:
        """Ход на доске, обновление статуса партии."""
        self.board.make_move(a)
        self.move_count += 1

        outcome = self.board.outcome()
        if outcome is None:
            return

        self.finished = True
        self._outcome = outcome if self.move_count % 2 == 1 else -outcome

    def outcome(self) -> int | None:
        """Итог с точки зрения первого: +1/0/-1."""
        return self._outcome
