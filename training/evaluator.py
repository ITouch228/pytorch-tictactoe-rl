import random

from agent.agent import Agent
from game.game import Game
from game.minimax import minimax_move
from utils.scores import side_outcome, win_draw_loss


class MinimaxEvaluator:
    """Оценка силы: сеть против минимакса на 9 дебютах."""

    def __init__(self, agent: Agent, seed: int = 0):
        self.agent = agent
        self.rng = random.Random(seed)

    def _play(self, opening: int, agent_as_x: bool) -> int:
        game = Game()
        game.step(opening)
        while not game.is_finished():
            if game.is_agent_turn(agent_as_x):
                a, _ = self.agent.select_action(game.state(), game.legal_moves(), greedy=True)
            else:
                a = minimax_move(game.board, rng=self.rng)
            game.step(a)
        outcome = game.outcome()
        if outcome is None:
            raise ValueError('Game is not finished')
        return outcome

    def stats(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Доли побед/ничьих/поражений агента за X и за O (по 9 дебютам)."""
        as_x, as_o = [], []
        for opening in range(9):
            as_x.append(side_outcome(self._play(opening, True), True))
            as_o.append(side_outcome(self._play(opening, False), False))
        return win_draw_loss(as_x), win_draw_loss(as_o)
