import random

from game.board import Board

# Значения всего дерева игры кешируются на весь процесс.
_VALUE_CACHE: dict[Board, int] = {}


def minimax_value(board: Board) -> int:
    """Возвращает значение доски для того, чей ход (negamax + мемоизация).

    На терминальной доске outcome() возвращает результат с точки зрения
    последнего ходившего, а нам нужен с точки зрения «следующего» —
    поэтому знак инвертируется.
    """
    cached = _VALUE_CACHE.get(board)
    if cached is not None:
        return cached

    outcome = board.outcome()
    if outcome is not None:
        return -outcome

    best = -2
    for a in board.legal_moves():
        b = board.clone()
        b.make_move(a)
        score = -minimax_value(b)
        if score > best:
            best = score
    _VALUE_CACHE[board] = best
    return best


def minimax_move(board: Board, rng=None) -> int:
    """Возвращает лучший ход по минимаксу.

    При равенстве очков выбирает случайно из равносильных, иначе
    все партии против минимакса были бы идентичными.
    """
    best_score, best_actions = -2, []
    for a in board.legal_moves():
        b = board.clone()
        b.make_move(a)
        score = -minimax_value(b)
        if score > best_score:
            best_score, best_actions = score, [a]
        elif score == best_score:
            best_actions.append(a)
    return (rng or random).choice(best_actions)
