"""Загрузка обученной сети и создание Agent из Config.

Единая точка загрузки checkpoint'а, чтобы скрипты не дублировали
QNet/load_state_dict/eval и не хардкодили hidden/save_path.
"""

import torch

from agent.agent import Agent
from model.qnet import QNet


def load_agent(cfg) -> Agent:
    net = QNet(hidden=cfg.hidden)
    net.load_state_dict(torch.load(cfg.save_path, map_location='cpu', weights_only=True))
    net.eval()
    return Agent(
        net,
        eps_start=cfg.eps_start,
        eps_end=cfg.eps_end,
        eps_decay=cfg.eps_decay,
    )
