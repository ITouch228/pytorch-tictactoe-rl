"""Перевод исходов между перспективами и подсчёт win/draw/loss."""


def side_outcome(outcome: int, agent_as_x: bool) -> int:
    """Исход партии (+1/0/-1) с точки зрения агента, а не первого игрока."""
    return outcome if agent_as_x else -outcome


def target_for_move(outcome: int, move_index: int) -> float:
    """MC-мишень ходу: исход с точки зрения того, кто сделал этот ход."""
    return float(outcome if move_index % 2 == 0 else -outcome)


def win_draw_loss(outcomes: list[int]) -> tuple[float, float, float]:
    """Доли побед/ничьих/поражений по списку исходов (+1/0/-1)."""
    n = len(outcomes)
    if n == 0:
        return 0.0, 0.0, 0.0
    wins = sum(o == 1 for o in outcomes)
    draws = sum(o == 0 for o in outcomes)
    losses = sum(o == -1 for o in outcomes)
    return wins / n, draws / n, losses / n
