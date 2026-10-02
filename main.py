import time

import torch

from agent.agent import Agent
from config import Config
from model.qnet import QNet
from training.buffer import ReplayBuffer
from training.evaluator import MinimaxEvaluator
from training.trainer import Trainer
from utils.benchmark import append_benchmark
from utils.seed import set_seed
from viz.visualizer import Visualizer


def main():
    cfg = Config()
    set_seed(cfg.seed)
    start_time = time.time()

    net = QNet(hidden=cfg.hidden)
    agent = Agent(
        net,
        eps_start=cfg.eps_start,
        eps_end=cfg.eps_end,
        eps_decay=cfg.eps_decay,
    )
    buffer = ReplayBuffer(capacity=cfg.buffer_capacity)
    trainer = Trainer(
        net,
        agent,
        buffer,
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        train_steps_per_episode=cfg.train_steps_per_episode,
        min_buffer_size=cfg.min_buffer_size,
        n_random_opening=cfg.n_random_opening,
    )
    evaluator = MinimaxEvaluator(agent)
    viz = Visualizer(
        net,
        evaluator,
        update_every=cfg.viz_every,
        hidden_shown=cfg.hidden_shown,
        power=1.0,
        demo_delay=cfg.viz_speed,
    )

    best_draws = -1.0
    converged_ep = None  # первый аудит с draws=1.000 - скорость сходимости

    def audit(episode):
        """Оценка против эталона, при рекорде сохраняем в save_path"""
        nonlocal best_draws, converged_ep
        (w1, d1, l1), (w2, d2, l2) = evaluator.stats()
        print(f'[{episode}] X vs minimax: wins={w1:.3f} draws={d1:.3f} losses={l1:.3f}')
        print(f'[{episode}] O vs minimax: wins={w2:.3f} draws={d2:.3f} losses={l2:.3f}')
        if d1 >= 1.0 and d2 >= 1.0 and converged_ep is None:
            converged_ep = episode
        if (d1 + d2) / 2 > best_draws:
            best_draws = (d1 + d2) / 2
            torch.save(net.state_dict(), cfg.save_path)
            print(f'[{episode}] record (draws={(d1 + d2) / 2:.3f}) — saved {cfg.save_path}')
        return (d1 + d2) / 2

    def on_episode_end(episode, trainer):
        viz.update(trainer, episode, agent.epsilon(episode))
        if episode > 0 and episode % cfg.audit_every == 0:
            print(f'[{episode}] noise: {trainer.noise_rate():.3f}')
            audit(episode)

    trainer.run(
        num_episodes=cfg.episodes,
        on_episode_end=on_episode_end,
        vs_minimax_every=cfg.vs_minimax_every,
    )
    final_draws = audit(cfg.episodes - 1)

    minutes = (time.time() - start_time) / 60
    append_benchmark(
        cfg,
        {
            'date': time.strftime('%Y-%m-%d %H:%M'),
            'run': cfg.run_name,
            'seed': cfg.seed,
            'episodes': cfg.episodes,
            'lr': cfg.lr,
            'batch': cfg.batch_size,
            'steps': cfg.train_steps_per_episode,
            'eps_decay': cfg.eps_decay,
            'eps_end': cfg.eps_end,
            'n_random_opening': cfg.n_random_opening,
            'vs_minimax_every': cfg.vs_minimax_every,
            'min_buffer_size': cfg.min_buffer_size,
            'buffer': cfg.buffer_capacity,
            'converged_ep': '' if converged_ep is None else converged_ep,
            'best_draws': f'{best_draws:.3f}',
            'final_draws': f'{final_draws:.3f}',
            'final_noise': f'{trainer.noise_rate():.3f}',
            'minutes': f'{minutes:.1f}',
        },
    )
    print(f'\n=== ИТОГ: {cfg.run_name} ===')
    converged_str = f'ep {converged_ep}' if converged_ep is not None else 'не сошлась'
    print(f'сходимость (draws=1.000 vs minimax): {converged_str}')
    print(f'итог: best_draws={best_draws:.3f}, final_draws={final_draws:.3f} за {minutes:.1f} мин')
    print(f'строка добавлена в {cfg.benchmark_path}')
    viz.close()


if __name__ == '__main__':
    main()
