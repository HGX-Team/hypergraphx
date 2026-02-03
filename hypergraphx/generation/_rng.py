from __future__ import annotations

import random
from typing import Optional

import numpy as np


def np_rng(seed: Optional[int] = None) -> np.random.Generator:
    return np.random.default_rng(seed)


def py_rng(seed: Optional[int] = None) -> random.Random:
    return random.Random(seed)


def split_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 2**32 - 1, dtype=np.uint32))
