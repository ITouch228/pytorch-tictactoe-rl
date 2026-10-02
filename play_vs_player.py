import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import torch
from matplotlib.widgets import Button

from config import Config
from game.game import Game
from utils.absolute_board import AbsoluteBoard
from utils.agent_loader import load_agent
from viz.visualizer import BoardRenderer, NetworkRenderer


class GameUI:
    """Окно игры: доска слева (кликабельная), граф сети справа.

    Обработчик привязан напрямую к Tk-виджету, поэтому ловит и
    клики мышью, и тапы на тачпаде (tap-to-click).
    """

    def __init__(self, agent, human_is_x=True):
        self.agent = agent
        self.human_is_x = human_is_x
        self.game = Game()

        # абсолютная доска для отрисовки: X=+1, O=-1.
        self.absolute_board = AbsoluteBoard()

        self.waiting_for_net = False
        self.finished = False

        # --- фигура ---
        self.fig = plt.figure(figsize=(13, 6))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1, 1.4], wspace=0.15)
        self.ax_board = self.fig.add_subplot(gs[0, 0])
        self.ax_net = self.fig.add_subplot(gs[0, 1])

        self.board_renderer = BoardRenderer(self.ax_board)
        self.net_renderer = NetworkRenderer(
            self.ax_net,
            agent.net,
            hidden_shown=64,
            power=2.0,
        )

        self.ax_btn = self.fig.add_axes((0.02, 0.02, 0.09, 0.07))
        self.btn_restart = Button(self.ax_btn, 'Заново')
        self.btn_restart.on_clicked(self._restart)

        self._set_turn_title()

        # Привязка к Tk напрямую — надёжно ловит тачпад.
        # Используем add='+' чтобы не переопределять обработчики matplotlib.
        self._tk_widget = self.fig.canvas.get_tk_widget()
        self._tk_widget.bind('<Button-1>', self._on_tk_click, add='+')
        self._tk_widget.bind('<ButtonRelease-1>', self._on_tk_click, add='+')

        self._refresh()

        if not human_is_x:
            self._net_move()

    # ---------- рестарт ----------

    def _set_turn_title(self):
        """Ставит заголовок доски с указанием стороны игрока."""
        side = 'X' if self.human_is_x else 'O'
        self.ax_board.set_title(f'Ты играешь за {side}')

    def _restart(self, event=None):
        """Сбрасывает партию по кнопке «Заново»."""
        self.game = Game()
        self.absolute_board.reset()
        self.finished = False
        self._set_turn_title()
        self._refresh()
        if not self.human_is_x:
            self._net_move()

    # ---------- Tk-обработчик ----------

    def _on_tk_click(self, tk_event):
        """Обрабатывает клик по доске, конвертируя координаты Tk в клетку."""
        if self.finished or self.waiting_for_net:
            return
        # обрабатываем только нажатие, чтобы не срабатывать дважды
        if tk_event.type != '4':  # ButtonPress = '4'
            return

        px = float(tk_event.x)
        py = float(tk_event.y)

        # в Tk ось y направлена вниз, в matplotlib — вверх
        h = self._tk_widget.winfo_height()
        my = h - py

        bbox = self.ax_board.get_window_extent(self.fig.canvas)
        if not (bbox.x0 <= px <= bbox.x1 and bbox.y0 <= my <= bbox.y1):
            return

        inv = self.ax_board.transData.inverted()
        data_x, data_y = inv.transform((px, my))

        col = int(data_x)
        row = int(3 - data_y)
        if not (0 <= col < 3 and 0 <= row < 3):
            return
        a = row * 3 + col

        if a not in self.game.legal_moves():
            return

        self._human_move(a)
        if not self.finished:
            self._net_move()

    # ---------- ходы ----------

    def _human_move(self, a):
        """Применяет ход человека и обновляет доску."""
        self.game.step(a)
        self.absolute_board.place(a)
        self._check_finish()
        self._refresh()

    def _net_move(self):
        """Применяет жадный ход сети и обновляет доску."""
        self.waiting_for_net = True

        a, _ = self.agent.select_action(
            self.game.state(),
            self.game.legal_moves(),
            greedy=True,
        )
        self.game.step(a)
        self.absolute_board.place(a)

        self.waiting_for_net = False
        self._check_finish()
        self._refresh()

    def _check_finish(self):
        """Если партия окончена — выводит её итог в заголовок доски."""
        if not self.game.is_finished():
            return
        self.finished = True
        r = self.game.outcome()
        human_result = r if self.human_is_x else -r
        if human_result == 1:
            self.ax_board.set_title('Ты выиграл!', color='green')
        elif human_result == -1:
            self.ax_board.set_title('Сеть выиграла', color='red')
        else:
            self.ax_board.set_title('Ничья', color='gray')

    # ---------- отрисовка ----------

    def _refresh(self):
        """Перерисовывает доску и граф сети по текущему состоянию."""
        self.board_renderer.draw(self.absolute_board.cells)

        state = self.game.state()
        with torch.no_grad():
            h1, h2 = self.agent.net.activations(torch.tensor(state, dtype=torch.float32))
        self.net_renderer.update(self.agent.net, (h1.numpy(), h2.numpy()))

        self.fig.canvas.draw()


def main():
    cfg = Config()
    agent = load_agent(cfg)

    print('Игра против обученной сети.')
    print('  0 1 2')
    print('  3 4 5')
    print('  6 7 8')
    print()
    print('Кем играешь? X (первый) или O (второй):')
    side = input('> ').strip().lower()
    human_is_x = side != 'o'

    GameUI(agent, human_is_x=human_is_x)

    plt.show(block=True)


if __name__ == '__main__':
    main()
