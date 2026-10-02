import random
from collections import deque

import numpy as np
import torch

from agent.agent import Agent
from game.game import Game
from game.minimax import minimax_move
from model.qnet import QNet
from training.buffer import ReplayBuffer
from utils.batch import augment_batch
from utils.constants import SYMMETRIES
from utils.scores import side_outcome, target_for_move

# Скользящие окна метрик
_LOSS_WINDOW = 20_000
_NOISE_WINDOW = 20_000


class Trainer:
    """Self-play обучение через Monte Carlo.

    Играем партию, в конце проставляем каждому ходу target = исход игры
    (с точки зрения того, кто ходил), кладём в буфер, учимся на случайных
    батчах. Симметрии доски аугментируют батч ×8.
    """

    def __init__(
        self,
        net: QNet,
        agent: Agent,
        buffer: ReplayBuffer | None,
        lr: float,
        batch_size: int,
        train_steps_per_episode: int,
        min_buffer_size: int,
        n_random_opening: int,
    ):
        self.net = net
        self.agent = agent
        self.buffer = buffer if buffer is not None else ReplayBuffer()
        self.optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        self.batch_size = batch_size
        self.train_steps_per_episode = train_steps_per_episode
        self.min_buffer_size = min_buffer_size
        self.n_random_opening = n_random_opening

        self.losses: deque[float] = deque(maxlen=_LOSS_WINDOW)
        self.noise: deque[float] = deque(maxlen=_NOISE_WINDOW)

    def play_episode(self, episode: int, n_random_opening: int | None = None) -> int:
        """Self-play партия с рандомным дебютом.

        Первые n_random_opening ходов - случайные, не идут в буфер.
        """
        if n_random_opening is None:
            n_random_opening = self.n_random_opening
        game = Game()
        history = []  # [(state, action, was_random), ...]

        for _ in range(random.randint(0, n_random_opening)):
            if game.is_finished():
                break
            a = random.choice(game.legal_moves())
            history.append((game.state(), a, True))
            game.step(a)

        while not game.is_finished():
            state = game.state()
            legal = game.legal_moves()
            action, was_random = self.agent.select_action(state, legal, episode)
            history.append((state, action, was_random))
            game.step(action)

        outcome = game.outcome()
        assert outcome is not None, 'партия обязана быть доиграна'

        # эмпирическая доля случайных ходов в партии
        random_moves = sum(1 for _, _, r in history if r)
        self.noise.append(random_moves / len(history) if history else 0.0)

        for i, (state, action, was_random) in enumerate(history):
            if was_random:
                continue
            self.buffer.push(state, action, target_for_move(outcome, i))

        return outcome

    def train_step(self) -> float | None:
        """Один шаг обучения. Возвращает None если буфер ещё не наполнен."""
        if len(self.buffer) < self.min_buffer_size:
            return None

        batch = self.buffer.sample(self.batch_size)
        states = np.array([b[0] for b in batch], dtype=np.float32)
        actions = np.array([b[1] for b in batch], dtype=np.int64)
        targets = np.array([b[2] for b in batch], dtype=np.float32)

        # аугментация выполняется на numpy до конвертации в тензоры
        states, actions, targets = augment_batch(states, actions, targets, SYMMETRIES)

        states_t = torch.tensor(states, dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.long)
        targets_t = torch.tensor(targets, dtype=torch.float32)

        q_all = self.net(states_t)
        q_chosen = q_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        loss = ((q_chosen - targets_t) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.losses.append(loss.item())
        return loss.item()

    def play_vs_minimax(self, episode, agent_is_x=True):
        """Партия против минимакса — ходы обеих сторон в буфер."""
        game = Game()
        history = []
        agent_moves = 0
        agent_random = 0

        while not game.is_finished():
            state = game.state()
            if game.is_agent_turn(agent_is_x):
                action, was_random = self.agent.select_action(state, game.legal_moves(), episode)
                agent_moves += 1
                agent_random += int(was_random)
            else:
                action = minimax_move(game.board)
            history.append((state, action))
            game.step(action)

        outcome = game.outcome()
        if outcome is None:
            raise ValueError('Game is not finished')

        self.noise.append(agent_random / agent_moves if agent_moves else 0.0)

        for i, (state, action) in enumerate(history):
            self.buffer.push(state, action, target_for_move(outcome, i))

        return side_outcome(outcome, agent_is_x)

    def run(self, num_episodes, on_episode_end=None, vs_minimax_every=20):
        """Основной цикл обучения."""
        for ep in range(num_episodes):
            if ep > 0 and vs_minimax_every and ep % vs_minimax_every == 0:
                agent_is_x = (ep // vs_minimax_every) % 2 == 0
                self.play_vs_minimax(ep, agent_is_x=agent_is_x)
            else:
                self.play_episode(ep)
            for _ in range(self.train_steps_per_episode):
                self.train_step()
            if on_episode_end is not None:
                on_episode_end(ep, self)

    def noise_rate(self, last_n=500):
        """Средняя доля случайных ходов в последних партиях."""
        recent = list(self.noise)[-last_n:]
        return sum(recent) / len(recent) if recent else 0.0
