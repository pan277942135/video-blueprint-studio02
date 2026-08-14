import json
import shutil
import subprocess

import numpy as np
import pytest

from packages.pipeline_core.binary_rle import encode_rle
from packages.pipeline_core.point_tracks import run_point_track_refinement
from packages.pipeline_core.sparse_motion import SparseMotionConfig


def test_point_ids_do_not_cross_hard_shot_boundary(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is required")

    video = tmp_path / "motion.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=128x96:rate=10",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
        capture_output=True,
    )

    frame_count = 10
    height, width = 96, 128
    mask = np.ones((height, width), dtype=np.bool_)
    frames = [{"size": [height, width], "counts": encode_rle(mask)} for _ in range(frame_count)]
    mask_uri = "artifacts/timeseries/char_0001_person_mask.rle.json"
    mask_path = tmp_path / "person_mask.json"
    mask_path.write_text(
        json.dumps({"frame_count": frame_count, "height": height, "width": width, "frames": frames}),
        encoding="utf-8",
    )
    character = {"character_id": "char_0001", "person_mask_ref": {"uri": mask_uri}}

    _, emitted, _, _ = run_point_track_refinement(
        str(video),
        characters=[character],
        shots=[{"frame_start": 0}, {"frame_start": 5}],
        frame_count=frame_count,
        output_dir=str(tmp_path),
        sidecars={mask_uri: str(mask_path)},
        config=SparseMotionConfig(max_points=24),
    )

    track_uri = "artifacts/timeseries/char_0001_point_tracks.npz"
    with np.load(emitted[track_uri], allow_pickle=False) as bundle:
        track_ids = bundle["track_id"]

    before = set(int(value) for value in track_ids[4] if value >= 0)
    after = set(int(value) for value in track_ids[5] if value >= 0)
    assert before
    assert after
    assert before.isdisjoint(after)
    assert min(after) > max(before)
