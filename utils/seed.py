import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python's `random` and NumPy's global RNG.

    Both are seeded (not just NumPy) so that any incidental use of the
    standard-library `random` module elsewhere in the project stays
    deterministic too. Call once at the start of a script/test so results
    reproduce from a clean clone.
    """
    random.seed(seed)
    np.random.seed(seed)
