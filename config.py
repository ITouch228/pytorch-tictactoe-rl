class Config:
    """
    Параметры проекта.
    """

    version = 'v1.0'

    # рандом
    seed = 42
    eval_seed = 0

    # сеть
    hidden = 64

    # обучение
    episodes = 6_001
    lr = 5e-4
    batch_size = 64
    train_steps_per_episode = 16
    min_buffer_size = 512
    buffer_capacity = 50_000
    n_random_opening = 0  # сколько первых ходов self-play может играть случайно

    # исследование
    eps_start = 1.0
    eps_end = 0.0
    eps_decay = 2_000

    # визуализация
    hidden_shown = 64
    viz_every = 200
    viz_speed = 0.00

    # аудит против минимакса и учебные партии против эталона
    audit_every = 200
    vs_minimax_every = 0

    # сохранение
    save_path = 'qnet.pt'
    benchmark_path = 'benchmark.csv'

    run_name = f'{version}'
