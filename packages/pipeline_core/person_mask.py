from __future__ import annotations

from typing import Any

import numpy as np

from packages.pipeline_core.binary_rle import decode_rle, encode_rle
from packages.pipeline_core.person_mask_core import (
    PersonMaskError,
    PersonMaskObservation,
    PersonMaskSegmenter,
    run_person_mask_refinement,
)


def encode_binary_rle(mask: np.ndarray) -> dict[str, Any]:
    """Encode a 2D binary mask using the canonical row-major RLE codec."""
    values = np.asarray(mask, dtype=np.bool_)
    if values.ndim != 2:
        raise ValueError("binary mask must be 2D")
    height, width = values.shape
    return {
        "height": int(height),
        "width": int(width),
        "counts": encode_rle(values),
    }


def decode_binary_rle(payload: dict[str, Any]) -> np.ndarray:
    """Decode canonical/legacy person-mask RLE envelopes without changing codec semantics."""
    if not isinstance(payload, dict):
        raise ValueError("RLE payload must be an object")
    counts = payload.get("counts")
    if not isinstance(counts, list) or not all(isinstance(value, int) for value in counts):
        raise ValueError("RLE counts must be an integer array")

    height = payload.get("height")
    width = payload.get("width")
    if not isinstance(height, int) or not isinstance(width, int):
        size = payload.get("size")
        if not (
            isinstance(size, list)
            and len(size) == 2
            and all(isinstance(value, int) for value in size)
        ):
            raise ValueError("RLE payload must declare height/width or size")
        height, width = size
    return decode_rle(counts, height, width)


__all__ = [
    "PersonMaskError",
    "PersonMaskObservation",
    "PersonMaskSegmenter",
    "decode_binary_rle",
    "encode_binary_rle",
    "run_person_mask_refinement",
]
