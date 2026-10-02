import random
from collections import deque

import numpy as np


class ReplayBuffer:
    """Кольцевой буфер переходов (state, action, target)."""

    def __init__(self, capacity: int = 30000) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, target: float) -> None:
        self.buffer.append((state, action, target))

    def sample(self, batch_size: int) -> list[tuple[np.ndarray, int, float]]:
        """Возвращает случайную выборку переходов размером batch_size."""
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)
