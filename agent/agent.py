import random

import numpy as np
import torch

from model.qnet import QNet


class Agent:
    """Выбирает ход из Q-значений сети: ε-greedy + маска нелегальных клеток."""

    def __init__(self, net: QNet, eps_start: float, eps_end: float, eps_decay: int) -> None:
        self.net = net
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

    def epsilon(self, episode: int) -> float:
        """Текущий exploration: исследование экспоненциально затухает к eps_end."""
        return self.eps_end + (self.eps_start - self.eps_end) * np.exp(-episode / self.eps_decay)

    @torch.no_grad()
    def q_values(self, state: np.ndarray) -> np.ndarray:
        """Возвращает Q для всех 9 клеток без маски. state - np.ndarray[9]."""
        x = torch.tensor(state, dtype=torch.float32)
        return self.net(x).numpy()

    def mask_illegal(self, q: np.ndarray, legal: list[int]) -> np.ndarray:
        """Заменяет Q нелегальных ходов на -inf."""
        masked = np.full_like(q, -np.inf)
        masked[legal] = q[legal]
        return masked

    def select_action(
        self,
        state: np.ndarray,
        legal: list[int],
        episode: int | None = None,
        greedy: bool = False,
    ) -> tuple[int, bool]:
        """
        Выбирает ход из легальных. greedy=True отключает исследование.

        Возвращает (action, was_random)
        """
        if not greedy and random.random() < self.epsilon(episode or 0):
            return random.choice(legal), True

        q = self.mask_illegal(self.q_values(state), legal)
        return int(np.argmax(q)), False
