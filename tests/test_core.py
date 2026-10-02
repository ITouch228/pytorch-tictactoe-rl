import numpy as np
import pytest

from agent.agent import Agent
from game.board import Board
from game.game import Game
from game.minimax import minimax_move
from model.qnet import QNet
from training.buffer import ReplayBuffer
from training.evaluator import MinimaxEvaluator
from utils.scores import side_outcome, target_for_move, win_draw_loss


def _agent() -> Agent:
    net = QNet(hidden=16)
    net.eval()
    return Agent(net, eps_start=0.0, eps_end=0.0, eps_decay=1)


def test_buffer_len():
    b = ReplayBuffer(capacity=4)
    assert len(b) == 0
    b.push(np.zeros(9, dtype=np.float32), 0, 1.0)
    assert len(b) == 1


def test_board_rejects_illegal():
    board = Board()
    board.make_move(0)
    with pytest.raises(ValueError):
        board.make_move(0)  # клетка занята
    with pytest.raises(ValueError):
        board.make_move(9)  # вне доски


def test_step_returns_none():
    game = Game()
    assert game.step(4) is None
    assert game.move_count == 1


def test_game_is_agent_turn():
    game = Game()
    assert game.is_agent_turn(agent_as_x=True) is True
    assert game.is_agent_turn(agent_as_x=False) is False
    game.step(4)
    assert game.is_agent_turn(agent_as_x=True) is False


def test_scores_helpers():
    assert side_outcome(1, True) == 1
    assert side_outcome(1, False) == -1
    # метка = перспектива ходящего, agent не влияет:
    # ход 0 делал X (X->+1), метка +1
    assert target_for_move(1, 0) == 1.0
    # ход 1 делал O (O->-1), метка -1
    assert target_for_move(1, 1) == -1.0
    assert win_draw_loss([1, 0, -1, 1]) == (0.5, 0.25, 0.25)


def test_minimax_never_loses():
    # минимакс против себя с обеих сторон -> только ничья
    game = Game()
    while not game.is_finished():
        game.step(minimax_move(game.board))
    assert game.outcome() == 0


def test_evaluator_handles_draws():
    # сеть-заглушка: stats() обязан отработать без исключения даже при ничьих
    (wx, dx, lx), (wo, do, lo) = MinimaxEvaluator(_agent(), seed=0).stats()
    assert abs((wx + dx + lx) - 1.0) < 1e-6
    assert abs((wo + do + lo) - 1.0) < 1e-6
