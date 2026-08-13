import os
import shutil
import subprocess

import numpy as np
import pytest

from packages.pipeline_core.pose_estimation import PoseObservation, run_pose_estimation


class _FakePoseEstimator:
    def estimate(self, frame, bbox_xyxy, frame_idx, character_id):
        x1, y1, x2, y2 = bbox_xyxy
        xs = np.linspace(x1, x2, 17, dtype=np.float32)
        ys = np.linspace(y1, y2, 17, dtype=np.float32)
        return PoseObservation(
            frame_idx=frame_idx,
            character_id=character_id,
            keypoints_xy=np.stack([xs, ys], axis=1),
            confidence=np.full((17,), 0.8, dtype=np.float32),
        )


@pytest.fixture()
def pose_video(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is required for pose estimation test")
    video_path = tmp_path / "pose_source.mp4"
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


def _tracking_fixture(tmp_path, frame_count=5):
    bbox = np.full((frame_count, 4), np.nan, dtype=np.float32)
    bbox[0] = (20, 10, 120, 110)
    bbox[1] = (20, 10, 120, 110)
    bbox[3] = (22, 10, 122, 110)
    bbox[4] = (22, 10, 122, 110)
    uri = "artifacts/timeseries/char_000_tracking.npz"
    path = tmp_path / "tracking.npz"
    np.savez_compressed(path, bbox_xyxy=bbox)
    character = {
        "character_id": "char_000",
        "bbox_ref": {
            "uri": uri,
            "metadata": {"array_key": "bbox_xyxy"},
        },
    }
    return character, {uri: os.fspath(path)}


def test_pose_estimation_preserves_missing_frames_and_emits_sidecars(tmp_path, pose_video):
    character, tracking_sidecars = _tracking_fixture(tmp_path)
    characters, pose_sidecars, report_ref = run_pose_estimation(
        str(pose_video),
        characters=[character],
        frame_count=5,
        estimator=_FakePoseEstimator(),
        output_dir=str(tmp_path),
        sidecars=tracking_sidecars,
    )

    assert len(characters) == 1
    pose = characters[0]["pose"]
    assert pose["enabled"] is True
    assert pose["skeleton_name"] == "coco17"
    assert pose["keypoint_count"] == 17
    assert pose["quality"]["coverage"] == 1.0

    uri = pose["keypoints_2d_ref"]["uri"]
    assert uri in pose_sidecars
    with np.load(pose_sidecars[uri], allow_pickle=False) as bundle:
        keypoints = bundle["keypoints_xy"]
        confidence = bundle["confidence"]
    assert keypoints.shape == (5, 17, 2)
    assert confidence.shape == (5, 17)
    assert np.isnan(keypoints[2]).all()
    assert np.all(confidence[2] == 0.0)
    assert np.isfinite(keypoints[[0, 1, 3, 4]]).all()
    assert np.allclose(confidence[[0, 1, 3, 4]], 0.8)

    assert report_ref["kind"] == "pose_2d"
    assert report_ref["uri"] in pose_sidecars
    report = pose_sidecars[report_ref["uri"]]
    assert report["characters"][0]["present_frames"] == 4
    assert report["characters"][0]["estimated_frames"] == 4
    assert report["identity_inference_performed"] is False
    assert report["biometric_embedding_exported"] is False
