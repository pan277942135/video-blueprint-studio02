from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from packages.pipeline_core.binary_rle import decode_rle, encode_rle


class PersonMaskError(RuntimeError):
    pass


@dataclass(frozen=True)
class PersonMaskObservation:
    frame_idx: int
    character_id: str
    mask: np.ndarray
    confidence: float | None = None


class PersonMaskSegmenter(Protocol):
    def segment(
        self,
        frame: np.ndarray,
        bbox_xyxy: tuple[float, float, float, float],
        frame_idx: int,
        character_id: str,
    ) -> PersonMaskObservation | None: ...


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    data = json.dumps(value, indent=2, sort_keys=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _atomic_json(path: str, value: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temporary = f"{path}.tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _load_bboxes(character: dict[str, Any], frame_count: int, sidecars: dict[str, Any]) -> np.ndarray:
    character_id = str(character["character_id"])
    ref = character.get("bbox_ref")
    if not isinstance(ref, dict):
        raise PersonMaskError(f"{character_id} is missing bbox_ref")
    uri = ref.get("uri")
    key = ref.get("metadata", {}).get("array_key")
    path = sidecars.get(uri) if isinstance(uri, str) else None
    if not isinstance(key, str) or not isinstance(path, (str, os.PathLike)) or not os.path.isfile(str(path)):
        raise PersonMaskError(f"{character_id} bbox sidecar is unavailable")
    with np.load(str(path), allow_pickle=False) as bundle:
        if key not in bundle:
            raise PersonMaskError(f"{character_id} bbox array is missing")
        values = np.asarray(bundle[key], dtype=np.float32)
    if values.shape != (frame_count, 4):
        raise PersonMaskError(f"{character_id} bbox shape {values.shape} is invalid")
    return values


def _bbox(row: np.ndarray, character_id: str, frame_idx: int) -> tuple[float, float, float, float] | None:
    if np.all(np.isnan(row)):
        return None
    if not np.all(np.isfinite(row)):
        raise PersonMaskError(f"{character_id} frame {frame_idx} bbox is partially missing")
    x1, y1, x2, y2 = (float(value) for value in row)
    if x2 <= x1 or y2 <= y1:
        raise PersonMaskError(f"{character_id} frame {frame_idx} bbox is invalid")
    return x1, y1, x2, y2


def _validated_mask(
    observation: PersonMaskObservation,
    character_id: str,
    frame_idx: int,
    height: int,
    width: int,
) -> tuple[np.ndarray, float | None]:
    if observation.character_id != character_id or observation.frame_idx != frame_idx:
        raise PersonMaskError("mask observation does not match its frame-aligned track")
    mask = np.asarray(observation.mask)
    if mask.shape != (height, width):
        raise PersonMaskError(f"mask shape {mask.shape} does not match {(height, width)}")
    if mask.dtype != np.bool_:
        if not np.all(np.isfinite(mask)):
            raise PersonMaskError("mask contains non-finite values")
        if not {int(value) for value in np.unique(mask)}.issubset({0, 1}):
            raise PersonMaskError("mask must be binary")
        mask = mask.astype(np.bool_)
    if not np.any(mask):
        raise PersonMaskError("reported mask is empty")
    confidence = observation.confidence
    if confidence is not None:
        confidence = float(confidence)
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise PersonMaskError("confidence must be within [0, 1]")
    return mask, confidence


def _ref(uri: str, checksum: str, frame_count: int, height: int, width: int, coverage: float) -> dict[str, Any]:
    return {
        "uri": uri,
        "format": "rle_json",
        "dtype": "bool",
        "shape": [frame_count, height, width],
        "axes": ["frame", "y", "x"],
        "unit": "binary",
        "coordinate_space": "pixel_xy",
        "sampling": "per_frame",
        "frame_start": 0,
        "frame_end": frame_count - 1,
        "compression": None,
        "nan_policy": "preserve",
        "interpolation_policy": "none",
        "checksum_sha256": checksum,
        "metadata": {
            "encoding": "row_major_binary_rle_v1",
            "foreground_value": 1,
            "background_value": 0,
            "missing_frame_value": None,
            "coverage": round(coverage, 6),
        },
    }


def run_person_mask_refinement(
    video_path: str,
    *,
    characters: list[dict[str, Any]],
    frame_count: int,
    segmenter: PersonMaskSegmenter,
    output_dir: str,
    sidecars: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not os.path.isfile(video_path) or frame_count <= 0:
        raise PersonMaskError("valid normalized video and positive frame_count are required")

    bboxes = {str(c["character_id"]): _load_bboxes(c, frame_count, sidecars) for c in characters}
    frames: dict[str, list[dict[str, Any] | None]] = {
        character_id: [None] * frame_count for character_id in bboxes
    }
    present = {character_id: 0 for character_id in bboxes}
    observed = {character_id: 0 for character_id in bboxes}
    scores: dict[str, list[float]] = {character_id: [] for character_id in bboxes}

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise PersonMaskError("normalized video could not be opened")
    decoded = 0
    height = width = 0
    try:
        while decoded < frame_count:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            current_height, current_width = frame.shape[:2]
            if decoded == 0:
                height, width = current_height, current_width
            elif (current_height, current_width) != (height, width):
                raise PersonMaskError("normalized frame dimensions changed")
            for character in characters:
                character_id = str(character["character_id"])
                box = _bbox(bboxes[character_id][decoded], character_id, decoded)
                if box is None:
                    continue
                present[character_id] += 1
                result = segmenter.segment(frame, box, decoded, character_id)
                if result is None:
                    continue
                mask, confidence = _validated_mask(result, character_id, decoded, height, width)
                counts = encode_rle(mask)
                if not np.array_equal(decode_rle(counts, height, width), mask):
                    raise PersonMaskError("RLE round-trip failed")
                item: dict[str, Any] = {"size": [height, width], "counts": counts}
                if confidence is not None:
                    item["confidence"] = round(confidence, 8)
                    scores[character_id].append(confidence)
                frames[character_id][decoded] = item
                observed[character_id] += 1
            decoded += 1
    finally:
        capture.release()
    if decoded != frame_count:
        raise PersonMaskError(f"decoded {decoded} frames, expected {frame_count}")

    emitted: dict[str, Any] = {}
    report_rows: list[dict[str, Any]] = []
    coverages: list[float] = []
    measured_scores: list[float] = []
    for character in characters:
        character_id = str(character["character_id"])
        coverage = observed[character_id] / present[character_id] if present[character_id] else 0.0
        coverages.append(coverage)
        mean_score = float(np.mean(scores[character_id])) if scores[character_id] else None
        if mean_score is not None:
            measured_scores.append(mean_score)
        if observed[character_id] == 0:
            character["person_mask_ref"] = None
        else:
            uri = f"artifacts/timeseries/{character_id}_person_mask.rle.json"
            path = os.path.join(output_dir, uri.replace("/", os.sep))
            payload = {
                "schema": "vbs.person_mask.rle.v1",
                "character_id": character_id,
                "frame_count": frame_count,
                "height": height,
                "width": width,
                "encoding": "row_major_binary_rle_v1",
                "frames": frames[character_id],
            }
            _atomic_json(path, payload)
            checksum = _sha256_file(path)
            emitted[uri] = path
            character["person_mask_ref"] = _ref(uri, checksum, frame_count, height, width, coverage)
        report_rows.append(
            {
                "character_id": character_id,
                "track_present_frames": present[character_id],
                "mask_observed_frames": observed[character_id],
                "coverage": round(coverage, 6),
                "mean_confidence": round(mean_score, 6) if mean_score is not None else None,
            }
        )

    report_uri = "artifacts/reports/person_mask.json"
    report = {
        "stage": "person_mask",
        "frame_count": frame_count,
        "height": height,
        "width": width,
        "encoding": "row_major_binary_rle_v1",
        "interpolation": False,
        "characters": report_rows,
    }
    emitted[report_uri] = report
    report_ref = {
        "kind": "person_mask",
        "uri": report_uri,
        "sha256": _json_sha256(report),
        "mime_type": "application/json",
    }
    quality = {
        "coverage": round(min(coverages), 6) if coverages else 0.0,
        "score": round(min(measured_scores), 6) if measured_scores else 0.0,
        "confidence_available": bool(measured_scores),
    }
    return characters, emitted, report_ref, quality
