from config import Config
from training.evaluator import MinimaxEvaluator
from utils.agent_loader import load_agent


def main():
    cfg = Config()
    agent = load_agent(cfg)
    evaluator = MinimaxEvaluator(agent, seed=cfg.eval_seed)

    (wx, dx, lx), (wo, do, lo) = evaluator.stats()
    print(f'X wins: {wx:.3f}, draws: {dx:.3f}, losses: {lx:.3f}')
    print(f'O wins: {wo:.3f}, draws: {do:.3f}, losses: {lo:.3f}')


if __name__ == '__main__':
    main()
