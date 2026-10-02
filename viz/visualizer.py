import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

from model.qnet import QNet
from training.evaluator import MinimaxEvaluator
from utils.absolute_board import play_greedy_game


class BoardRenderer:
    """Рисует доску 3×3 с X и O на заданной оси."""

    def __init__(self, ax):
        self.ax = ax
        self.ax.set_xlim(0, 3)
        self.ax.set_ylim(0, 3)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.ax.set_title('Демо-партия (greedy)', fontsize=11)

        for i in range(1, 3):
            self.ax.plot([i, i], [0, 3], color='black', linewidth=1.5, alpha=0.3)
            self.ax.plot([0, 3], [i, i], color='black', linewidth=1.5, alpha=0.3)

        self.symbols = []
        for r in range(3):
            for c in range(3):
                t = self.ax.text(
                    c + 0.5, 2.5 - r, '', ha='center', va='center', fontsize=34, fontweight='bold'
                )
                self.symbols.append(t)

    def draw(self, cells):
        """Рисует фишки X/O по абсолютной доске cells (9 значений)."""
        for i, v in enumerate(cells):
            if v > 0:
                self.symbols[i].set_text('X')
                self.symbols[i].set_color('tab:blue')
            elif v < 0:
                self.symbols[i].set_text('O')
                self.symbols[i].set_color('tab:red')
            else:
                self.symbols[i].set_text('')


class NetworkRenderer:
    """Граф перцептрона: 9 → hidden → hidden → 9. Цвет и толщина линий по силе весов."""

    # синий — слабая, фиолетовый — средняя, красная — сильная
    CMAP = LinearSegmentedColormap.from_list('weak_strong', ['#2f6db4', '#8e44ad', '#c0392b'])

    def __init__(self, ax, net, hidden_shown=24, power=2.0):
        self.ax = ax
        self.ax.set_xlim(-0.2, 3.2)
        self.ax.set_ylim(-0.16, 1.05)
        self.ax.axis('off')
        self.ax.set_title(
            f'Сеть 9→{hidden_shown}→{hidden_shown}→9 (цвет и яркость = сила веса)',
            fontsize=11,
        )

        self.hidden_shown = hidden_shown
        self.power = power

        total_hidden = net.fc1.out_features
        rng = np.random.default_rng(42)
        self.h1_idx = np.sort(rng.choice(total_hidden, hidden_shown, replace=False))
        self.h2_idx = np.sort(rng.choice(total_hidden, hidden_shown, replace=False))

        def ypos(n, i):
            return (i + 0.5) / n

        self.pos_in = [(0.0, ypos(9, i)) for i in range(9)]
        self.pos_h1 = [(1.0, ypos(hidden_shown, i)) for i in range(hidden_shown)]
        self.pos_h2 = [(2.0, ypos(hidden_shown, i)) for i in range(hidden_shown)]
        self.pos_out = [(3.0, ypos(9, i)) for i in range(9)]

        def make_scatter(pos, color):
            xs = [p[0] for p in pos]
            ys = [p[1] for p in pos]
            # при полном скрытом слое точки мельче, иначе сливаются в полосу
            size = 140 if len(pos) <= 24 else 35
            return ax.scatter(
                xs,
                ys,
                s=size,
                c=color,
                edgecolors='black',
                linewidths=0.6,
                zorder=3,
            )

        self.sc_in = make_scatter(self.pos_in, 'lightsteelblue')
        self.sc_h1 = make_scatter(self.pos_h1, 'lightgray')
        self.sc_h2 = make_scatter(self.pos_h2, 'lightgray')
        self.sc_out = make_scatter(self.pos_out, 'lightcoral')

        for i in range(9):
            self.ax.text(
                -0.08, self.pos_in[i][1], str(i), ha='right', va='center', fontsize=8, color='gray'
            )
            self.ax.text(
                3.08, self.pos_out[i][1], str(i), ha='left', va='center', fontsize=8, color='gray'
            )

        self.lc1 = LineCollection([], zorder=1)
        self.lc2 = LineCollection([], zorder=1)
        self.lc3 = LineCollection([], zorder=1)
        ax.add_collection(self.lc1)
        ax.add_collection(self.lc2)
        ax.add_collection(self.lc3)

    def _style(self, w, wmax):
        """Цвет с alpha и толщина линии по силе веса."""
        mag = min(abs(w) / wmax, 1.0) if wmax > 0 else 0.0
        # power усиливает контраст: слабые веса тусклее сильных
        alpha = 0.07 + (mag**self.power) * 0.9
        width = 0.15 + 1.5 * mag
        r, g, b, _ = self.CMAP(mag)
        return (r, g, b, alpha), width, mag

    def _build_all(self, W, pos_from, pos_to, wmax):
        """Строит сегменты слоя, отсортированные по возрастанию силы."""
        items = []
        n_from, n_to = W.shape
        for i in range(n_from):
            for j in range(n_to):
                c, lw, mag = self._style(W[i, j], wmax)
                items.append((mag, [pos_from[i], pos_to[j]], c, lw))
        items.sort(key=lambda it: it[0])
        segs = [it[1] for it in items]
        colors = [it[2] for it in items]
        widths = [it[3] for it in items]
        return segs, colors, widths

    def update(self, net, activations=None):
        """Перестраивает связи по весам сети и подсвечивает активации."""
        W1 = net.fc1.weight.detach().numpy()
        W2 = net.fc2.weight.detach().numpy()
        W3 = net.fc3.weight.detach().numpy()

        W1_sub = W1[self.h1_idx, :].T  # [9, H]
        W2_sub = W2[self.h2_idx][:, self.h1_idx].T  # [H, H]
        W3_sub = W3[:, self.h2_idx].T  # [H, 9]

        # единая шкала силы для всех слоёв; 99-й перцентиль вместо
        # максимума — чтобы редкие выбросы не делали все связи тусклыми
        all_w = np.concatenate([W1_sub.ravel(), W2_sub.ravel(), W3_sub.ravel()])
        wmax = float(np.percentile(np.abs(all_w), 99))

        segs1, c1, w1 = self._build_all(W1_sub, self.pos_in, self.pos_h1, wmax)
        segs2, c2, w2 = self._build_all(W2_sub, self.pos_h1, self.pos_h2, wmax)
        segs3, c3, w3 = self._build_all(W3_sub, self.pos_h2, self.pos_out, wmax)

        self.lc1.set_segments(segs1)
        self.lc1.set_color(c1)
        self.lc1.set_linewidth(w1)
        self.lc2.set_segments(segs2)
        self.lc2.set_color(c2)
        self.lc2.set_linewidth(w2)
        self.lc3.set_segments(segs3)
        self.lc3.set_color(c3)
        self.lc3.set_linewidth(w3)

        if activations is not None:
            h1_act, h2_act = activations
            self.sc_h1.set_facecolor(self._act_colors(h1_act[self.h1_idx]))
            self.sc_h2.set_facecolor(self._act_colors(h2_act[self.h2_idx]))

    @staticmethod
    def _act_colors(acts):
        """Цвета нейронов по величине активации (шкала viridis)."""
        m = acts.max() if acts.max() > 0 else 1.0
        return [plt.cm.viridis(v / m) for v in acts]


class Visualizer:
    """Окно с доской, графом сети, loss, кривой силы и лог-строкой.

    Кривая силы — доля ничьих против минимакса (MinimaxEvaluator).
    """

    def __init__(
        self,
        net: QNet,
        evaluator: MinimaxEvaluator,
        update_every: int = 200,
        hidden_shown: int = 24,
        power: float = 2.0,
        demo_delay: float = 0.0,
    ):
        self.update_every = update_every
        self.demo_delay = demo_delay
        self.evaluator = evaluator

        plt.ion()
        self.fig = plt.figure(figsize=(16, 8))
        gs = self.fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.35, wspace=0.25)

        self.ax_board = self.fig.add_subplot(gs[0, 0])
        self.ax_net = self.fig.add_subplot(gs[0, 1])
        self.ax_loss = self.fig.add_subplot(gs[0, 2])
        self.ax_stats = self.fig.add_subplot(gs[1, :2])
        self.ax_log = self.fig.add_subplot(gs[1, 2])
        self.ax_log.axis('off')

        self.board_renderer = BoardRenderer(self.ax_board)
        self.net_renderer = NetworkRenderer(
            self.ax_net, net, hidden_shown=hidden_shown, power=power
        )

        (self.loss_line,) = self.ax_loss.plot([], [], 'b-')
        self.ax_loss.set_title('Loss (сглаженный)')
        self.ax_loss.set_xlabel('шаг обучения')
        self.ax_loss.set_ylim(0, 1.5)

        (self.draw1_line,) = self.ax_stats.plot([], [], 'r-', label='X (первый)')
        (self.draw2_line,) = self.ax_stats.plot([], [], 'b-', label='O (второй)')

        self.ax_stats.set_title('Статистика против минимакса (9 дебютов × обе стороны)')
        self.ax_stats.set_xlabel('эпизод')
        self.ax_stats.set_ylim(0, 1)
        self.ax_stats.legend(loc='lower right')

        self.log_text = self.ax_log.text(
            0.0, 1.0, '', fontsize=11, va='top', ha='left', family='monospace'
        )

        self.ep_history: list[int] = []
        self.draw1_history: list[float] = []
        self.draw2_history: list[float] = []

    @staticmethod
    def _moving_average(values, window):
        """Сглаживание скользящим средним по окну window."""
        if len(values) < window:
            return list(values)
        kernel = np.ones(window, dtype=np.float32) / window
        return list(np.convolve(np.array(values, dtype=np.float32), kernel, mode='valid'))

    def update(self, trainer, episode, eps):
        """Обновляет все панели раз в update_every эпизодов."""
        if episode % self.update_every != 0:
            return

        boards = play_greedy_game(trainer.agent)
        if self.demo_delay > 0:
            for b in boards:
                self.board_renderer.draw(b)
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
                plt.pause(self.demo_delay)
        else:
            self.board_renderer.draw(boards[-1])

        with torch.no_grad():
            h1, h2 = trainer.net.activations(torch.zeros(9))
        self.net_renderer.update(trainer.net, (h1.numpy(), h2.numpy()))

        if trainer.losses:
            smoothed = self._moving_average(list(trainer.losses), window=200)
            self.loss_line.set_data(range(len(smoothed)), smoothed)
            self.ax_loss.set_xlim(0, max(1, len(smoothed)))
            self.ax_loss.set_ylim(0, max(1.0, max(smoothed) * 1.1))

        (w1, d1, l1), (w2, d2, l2) = self.evaluator.stats()

        self.ep_history.append(episode)
        self.draw1_history.append(d1)
        self.draw2_history.append(d2)

        self.draw1_line.set_data(self.ep_history, self.draw1_history)
        self.draw2_line.set_data(self.ep_history, self.draw2_history)

        self.ax_stats.set_xlim(0, max(1, episode))

        cur_loss = trainer.losses[-1] if trainer.losses else 0.0
        self.log_text.set_text(
            f'эпизод   {episode}\n'
            f'eps      {eps:.3f}\n'
            f'noise    {trainer.noise_rate():.3f}\n'
            f'wins X    {w1:.3f}\n'
            f'wins O    {w2:.3f}\n'
            f'draws X    {d1:.3f}\n'
            f'draws O    {d2:.3f}\n'
            f'loss X    {l1:.3f}\n'
            f'loss O    {l2:.3f}\n'
            f'loss     {cur_loss:.4f}\n'
            f'buffer   {len(trainer.buffer)}'
        )

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        """Выключает интерактивный режим и показывает окно финально."""
        plt.ioff()
        plt.show()
