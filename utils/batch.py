import numpy as np


def augment_batch(states, actions, targets, symmetries):
    """Размножает батч симметриями доски.

    states: [B, 9], actions: [B], targets: [B] ->
    [B*K, 9], [B*K], [B*K], где K = len(symmetries).
    """
    B, K = states.shape[0], len(symmetries)  # noqa: N806
    aug_states = np.empty((B * K, 9), dtype=np.float32)
    aug_actions = np.empty(B * K, dtype=np.int64)
    aug_targets = np.empty(B * K, dtype=np.float32)

    idx = 0
    for i in range(B):
        s, a, t = states[i], int(actions[i]), targets[i]
        for perm in symmetries:
            # new[i] = old[perm[i]], а action переезжает в perm.index(a)
            aug_states[idx] = s[perm]
            aug_actions[idx] = perm.index(a)
            aug_targets[idx] = t
            idx += 1

    return aug_states, aug_actions, aug_targets
