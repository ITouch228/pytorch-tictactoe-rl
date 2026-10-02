import random

import numpy as np
import torch


def set_seed(seed):
    """
    Фиксирует сиды всех генераторов случайности.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
