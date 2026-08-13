import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from packages.pipeline_core.binary_rle import decode_rle, encode_rle
from packages.pipeline_core.person_mask import (
    PersonMaskError,
    PersonMaskObservation,
    run_person_mask_refinement,
)


def _write_video(path: Path, frame_count: int = 4, width: int = 64, height: int = 48) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 4.0, (width, height))
    assert writer.isOpened()
    try:
        for index in range(frame_count):
            frame = np.full((height, width, 3), 20 + index * 10, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def _tracking_fixture(tmp_path: Path) -> tuple[dict, dict[str, str]]:
    bboxes = np.array(
        [
            [10.0, 8.0, 30.0, 32.0],
            [np.nan, np.nan, np.nan, np.nan],
            [12.0, 8.0, 32.0, 32.0],
            [14.0, 8.0, 34.0, 32.0],
        ],
        dtype=np.float32,
    )
    path = tmp_path / "tracking.npz"
    np.savez_compressed(path, bbox_xyxy=bboxes)
    uri = "artifacts/timeseries/char_000_tracking.npz"
    character = {
        "character_id": "char_000",
        "bbox_ref": {"uri": uri, "metadata": {"array_key": "bbox_xyxy"}},
        "person_mask_ref": None,
    }
    return character, {uri: str(path)}


class _Segmenter:
    def segment(self, frame, bbox_xyxy, frame_idx, character_id):
        if frame_idx == 3:
            return None
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.bool_)
        x1, y1, x2, y2 = (round(value) for value in bbox_xyxy)
        mask[y1:y2, x1:x2] = True
        return PersonMaskObservation(frame_idx, character_id, mask, 0.8)


class _NoObservationSegmenter:
    def segment(self, frame, bbox_xyxy, frame_idx, character_id):
        return None


class _BadShapeSegmenter:
    def segment(self, frame, bbox_xyxy, frame_idx, character_id):
        return PersonMaskObservation(frame_idx, character_id, np.ones((2, 2), dtype=np.bool_), 0.5)


def test_binary_rle_round_trip() -> None:
    mask = np.array([[False, True, True], [False, False, True]], dtype=np.bool_)
    counts = encode_rle(mask)
    assert sum(counts) == mask.size
    assert np.array_equal(decode_rle(counts, 2, 3), mask)


def test_person_mask_exports_explicit_missing_frames(tmp_path: Path) -> None:
    video = tmp_path / "video.avi"
    _write_video(video)
    character, sidecars = _tracking_fixture(tmp_path)

    characters, emitted, report_ref, quality = run_person_mask_refinement(
        str(video),
        characters=[character],
        frame_count=4,
        segmenter=_Segmenter(),
        output_dir=str(tmp_path),
        sidecars=sidecars,
    )

    ref = characters[0]["person_mask_ref"]
    assert ref["format"] == "rle_json"
    assert ref["shape"] == [4, 48, 64]
    assert ref["interpolation_policy"] == "none"
    assert quality["coverage"] == pytest.approx(2 / 3, abs=1e-6)
    assert quality["score"] == pytest.approx(0.8, abs=1e-6)
    assert report_ref["kind"] == "person_mask"

    payload_path = emitted[ref["uri"]]
    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    assert payload["frames"][0] is not None
    assert payload["frames"][1] is None
    assert payload["frames"][2] is not None
    assert payload["frames"][3] is None
    decoded = decode_rle(payload["frames"][0]["counts"], 48, 64)
    assert decoded.dtype == np.bool_
    assert np.any(decoded)


def test_zero_observations_keep_null_ref_and_no_orphan_mask_file(tmp_path: Path) -> None:
    video = tmp_path / "video.avi"
    _write_video(video)
    character, sidecars = _tracking_fixture(tmp_path)
    characters, emitted, _, _ = run_person_mask_refinement(
        str(video),
        characters=[character],
        frame_count=4,
        segmenter=_NoObservationSegmenter(),
        output_dir=str(tmp_path),
        sidecars=sidecars,
    )
    assert characters[0]["person_mask_ref"] is None
    assert not any(key.endswith("_person_mask.rle.json") for key in emitted)


def test_wrong_mask_shape_fails_closed(tmp_path: Path) -> None:
    video = tmp_path / "video.avi"
    _write_video(video)
    character, sidecars = _tracking_fixture(tmp_path)
    with pytest.raises(PersonMaskError, match="mask shape"):
        run_person_mask_refinement(
            str(video),
            characters=[character],
            frame_count=4,
            segmenter=_BadShapeSegmenter(),
            output_dir=str(tmp_path),
            sidecars=sidecars,
        )
