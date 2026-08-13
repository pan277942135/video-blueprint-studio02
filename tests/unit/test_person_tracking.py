import os
import shutil
import subprocess

import numpy as np
import pytest

from packages.pipeline_core.person_tracking import (
    PersonDetection,
    PersonTrackingConfig,
    associate_anonymous_tracks,
    bbox_iou,
    run_person_tracking,
)


def _det(frame_idx: int, bbox=(10.0, 20.0, 110.0, 220.0), score=0.9):
    return PersonDetection(frame_idx=frame_idx, bbox_xyxy=bbox, score=score)


def test_bbox_iou():
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert bbox_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_anonymous_tracks_never_cross_shot_boundaries():
    shots = [
        {"shot_id": "shot_000", "index": 0, "frame_start": 0, "frame_end": 2},
        {"shot_id": "shot_001", "index": 1, "frame_start": 3, "frame_end": 5},
    ]
    detections = {frame: [_det(frame)] for frame in range(6)}

    tracks = associate_anonymous_tracks(shots, detections)

    assert len(tracks) == 2
    assert tracks[0].character_id == "char_000"
    assert tracks[0].shot_id == "shot_000"
    assert sorted(tracks[0].detections) == [0, 1, 2]
    assert tracks[1].character_id == "char_001"
    assert tracks[1].shot_id == "shot_001"
    assert sorted(tracks[1].detections) == [3, 4, 5]


def test_iou_association_is_deterministic_and_gap_bounded():
    shots = [{"shot_id": "shot_000", "index": 0, "frame_start": 0, "frame_end": 4}]
    detections = {
        0: [_det(0, (0, 0, 100, 100), 0.95)],
        1: [_det(1, (5, 0, 105, 100), 0.90)],
        4: [_det(4, (10, 0, 110, 100), 0.85)],
    }

    tracks = associate_anonymous_tracks(
        shots,
        detections,
        config=PersonTrackingConfig(iou_threshold=0.3, max_gap_frames=1),
    )

    assert len(tracks) == 2
    assert sorted(tracks[0].detections) == [0, 1]
    assert sorted(tracks[1].detections) == [4]


class _FakeDetector:
    def detect(self, frame: np.ndarray, frame_idx: int) -> list[PersonDetection]:
        height, width = frame.shape[:2]
        return [
            PersonDetection(
                frame_idx=frame_idx,
                bbox_xyxy=(width * 0.2, height * 0.1, width * 0.8, height * 0.9),
                score=0.8 + frame_idx * 0.01,
            )
        ]


@pytest.fixture()
def five_frame_video(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is required for person tracking tests")

    video_path = tmp_path / "five_frames.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=160x120:rate=5",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ],
        check=True,
        capture_output=True,
    )
    return video_path


def test_run_person_tracking_emits_dense_sidecars_and_overlay(five_frame_video, tmp_path):
    shots = [
        {
            "shot_id": "shot_000",
            "index": 0,
            "frame_start": 0,
            "frame_end": 4,
            "dominant_character_ids": [],
        }
    ]

    characters, sidecars, overlays, report_ref = run_person_tracking(
        str(five_frame_video),
        shots=shots,
        frame_count=5,
        detector=_FakeDetector(),
        output_dir=str(tmp_path / "artifacts_root"),
    )

    assert len(characters) == 1
    character = characters[0]
    assert character["character_id"] == "char_000"
    assert character["privacy"]["identity_inference_performed"] is False
    assert character["privacy"]["biometric_embedding_exported"] is False
    assert character["pose"]["enabled"] is False
    assert character["face"]["enabled"] is False
    assert character["face"]["landmarks_2d_ref"] is None
    assert character["face"]["landmarks_3d_ref"] is None
    assert character["presence"] == [{"frame_start": 0, "frame_end": 4}]
    assert shots[0]["dominant_character_ids"] == ["char_000"]

    tracking_uri = character["bbox_ref"]["uri"]
    assert tracking_uri in sidecars
    assert os.path.isfile(sidecars[tracking_uri])
    data = np.load(sidecars[tracking_uri])
    assert data["bbox_xyxy"].shape == (5, 4)
    assert data["visibility"].shape == (5,)
    assert data["pose_keypoints_2d"].shape == (5, 0, 3)

    assert len(overlays) == 1
    assert overlays[0]["uri"] in sidecars
    assert os.path.isfile(sidecars[overlays[0]["uri"]])
    assert report_ref["uri"] == "artifacts/reports/person_tracking.json"
    assert report_ref["uri"] in sidecars
    assert sidecars[report_ref["uri"]]["track_count"] == 1
