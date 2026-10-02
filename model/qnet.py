import torch
import torch.nn as nn


class QNet(nn.Module):
    """MLP-аппроксиматор Q(s, a): доска 9 -> 9 оценок ходов."""

    def __init__(self, hidden=64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(9, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 9)

    def _hidden(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        """Общие скрытые слои: x -> (h1, h2)."""
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        return h1, h2

    def forward(self, x) -> torch.Tensor:
        """x: [9] или [B, 9] -> Q: [9] или [B, 9]."""
        _, h2 = self._hidden(x)
        return self.fc3(h2)

    @torch.no_grad()
    def activations(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        """Скрытые активации для визуализации."""
        return self._hidden(x)
