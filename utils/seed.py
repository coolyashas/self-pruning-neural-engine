import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python's `random` and NumPy's global RNG for reproducibility.

    Call once at the start of a script/test.
    """
    random.seed(seed)
    np.random.seed(seed)
