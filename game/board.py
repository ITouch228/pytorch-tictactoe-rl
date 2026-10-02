from __future__ import annotations

import numpy as np

from utils.constants import LINES


class Board:
    """Состояние доски 3×3: perspective свои = +1, чужие = -1."""

    def __init__(self, cells: np.ndarray | list[int] | None = None) -> None:
        if cells is None:
            self.cells = np.zeros(9, dtype=np.float32)
        else:
            self.cells = np.array(cells, dtype=np.float32)

    def clone(self) -> Board:
        return Board(self.cells.copy())

    def legal_moves(self) -> list[int]:
        return [i for i in range(9) if self.cells[i] == 0]

    def is_full(self) -> bool:
        return bool(np.all(self.cells != 0))

    def outcome(self) -> int | None:
        """Итог с точки зрения последнего ходившего: +1/-1/0 или None."""
        for a, b, c in LINES:
            v = self.cells[a]
            if v != 0 and v == self.cells[b] == self.cells[c]:
                return int(v)
        if self.is_full():
            return 0
        return None

    def is_terminal(self) -> bool:
        return self.outcome() is not None

    def make_move(self, a: int) -> None:
        """Ставит фишку в клетку a и переворачивает перспективу."""
        if not (0 <= a < 9) or self.cells[a] != 0:
            raise ValueError(f'нелегальный ход: {a}')
        self.cells[a] = 1
        if not self.is_terminal():
            self.cells = -self.cells

    def to_array(self) -> np.ndarray:
        """Доска как float32-вектор длиной 9 для подачи в сеть."""
        return self.cells.astype(np.float32)

    def __str__(self) -> str:
        symbols = {1: 'X', -1: 'O', 0: '.'}
        rows = []
        for r in range(3):
            row = ' '.join(symbols[int(self.cells[r * 3 + c])] for c in range(3))
            rows.append(row)
        return '\n'.join(rows)

    __repr__ = __str__

    def __eq__(self, other) -> bool:
        return isinstance(other, Board) and np.array_equal(self.cells, other.cells)

    def __hash__(self) -> int:
        return hash(tuple(self.cells.tolist()))
