from __future__ import annotations

import numpy as np


def encode_rle(array: np.ndarray) -> list[int]:
    values = np.asarray(array, dtype=np.bool_)
    if values.ndim != 2:
        raise ValueError("array must be 2D")
    counts: list[int] = []
    current = False
    run = 0
    for raw in values.reshape(-1):
        value = bool(raw)
        if value == current:
            run += 1
        else:
            counts.append(run)
            run = 1
            current = value
    counts.append(run)
    return counts


def decode_rle(counts: list[int], height: int, width: int) -> np.ndarray:
    total = height * width
    if height <= 0 or width <= 0 or sum(counts) != total:
        raise ValueError("invalid RLE dimensions or counts")
    if any(count < 0 for count in counts):
        raise ValueError("RLE counts must be non-negative")
    flat = np.empty((total,), dtype=np.bool_)
    cursor = 0
    value = False
    for count in counts:
        flat[cursor : cursor + count] = value
        cursor += count
        value = not value
    return flat.reshape((height, width))
