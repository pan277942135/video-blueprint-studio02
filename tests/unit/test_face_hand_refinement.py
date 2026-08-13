import hashlib
import os
import shutil
import subprocess

import numpy as np
import pytest

from packages.pipeline_core.face_hand_refinement import (
    FACE_LANDMARK_COUNT,
    HAND_LANDMARK_COUNT,
    FaceHandObservation,
    FaceHandRefinementError,
    FaceObservation,
    HandObservation,
    run_face_hand_refinement,
)


def _tracking_character(tmp_path, frame_count: int = 3):
    uri = "artifacts/timeseries/char_000_tracking.npz"
    path = tmp_path / uri
    path.parent.mkdir(parents=True, exist_ok=True)
    bboxes = np.tile(np.array([[8.0, 8.0, 152.0, 112.0]], dtype=np.float32), (frame_count, 1))
    np.savez_compressed(path, bbox_xyxy=bboxes)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    character = {
        "character_id": "char_000",
        "bbox_ref": {
            "uri": uri,
            "format": "npz",
            "dtype": "float32",
            "shape": [frame_count, 4],
            "axes": ["frame", "bbox_component"],
            "unit": "px",
            "coordinate_space": "pixel_xy",
            "frame_start": 0,
            "frame_end": frame_count - 1,
            "checksum_sha256": checksum,
            "metadata": {"array_key": "bbox_xyxy"},
        },
        "privacy": {
            "identity_inference_performed": False,
            "biometric_embedding_exported": False,
        },
    }
    return character, {uri: str(path)}


class _FakeRefiner:
    def refine(self, frame, bbox_xyxy, frame_idx, character_id):
        face_x = np.linspace(30.0, 90.0, FACE_LANDMARK_COUNT, dtype=np.float32)
        face_y = np.linspace(20.0, 82.0, FACE_LANDMARK_COUNT, dtype=np.float32)
        face = FaceObservation(
            landmarks_xy=np.column_stack([face_x, face_y]),
            bbox_xyxy=(28.0, 18.0, 92.0, 84.0),
            confidence=0.91,
        )
        hands = ()
        if frame_idx == 1:
            hand_x = np.linspace(98.0, 124.0, HAND_LANDMARK_COUNT, dtype=np.float32)
            hand_y = np.linspace(54.0, 88.0, HAND_LANDMARK_COUNT, dtype=np.float32)
            hands = (
                HandObservation(
                    side="left",
                    landmarks_xy=np.column_stack([hand_x, hand_y]),
                    bbox_xyxy=(96.0, 52.0, 126.0, 90.0),
                    confidence=0.82,
                    handedness_confidence=0.96,
                ),
            )
        return FaceHandObservation(
            frame_idx=frame_idx,
            character_id=character_id,
            face=face,
            hands=hands,
        )


class _NoObservationRefiner:
    def refine(self, frame, bbox_xyxy, frame_idx, character_id):
        return None


class _InvalidFaceRefiner(_FakeRefiner):
    def refine(self, frame, bbox_xyxy, frame_idx, character_id):
        observation = super().refine(frame, bbox_xyxy, frame_idx, character_id)
        assert observation is not None and observation.face is not None
        return FaceHandObservation(
            frame_idx=frame_idx,
            character_id=character_id,
            face=FaceObservation(
                landmarks_xy=np.zeros((477, 2), dtype=np.float32),
                bbox_xyxy=observation.face.bbox_xyxy,
                confidence=observation.face.confidence,
            ),
            hands=(),
        )


@pytest.fixture()
def three_frame_video(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is required for face/hands refinement tests")
    video_path = tmp_path / "three_frames.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=160x120:rate=3",
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


def test_refinement_emits_frame_aligned_geometry_without_interpolation(three_frame_video, tmp_path):
    character, sidecars = _tracking_character(tmp_path)
    characters, refinement_sidecars, report_ref = run_face_hand_refinement(
        str(three_frame_video),
        characters=[character],
        frame_count=3,
        refiner=_FakeRefiner(),
        output_dir=str(tmp_path / "out"),
        sidecars=sidecars,
    )

    result = characters[0]
    assert result["privacy"]["identity_inference_performed"] is False
    assert result["privacy"]["biometric_embedding_exported"] is False
    assert result["face"]["enabled"] is True
    assert result["face"]["landmark_count"] == FACE_LANDMARK_COUNT
    assert result["face"]["landmarks_2d_ref"]["coordinate_space"] == "pixel_xy"
    assert result["face"]["landmarks_2d_ref"]["shape"] == [3, FACE_LANDMARK_COUNT, 2]
    assert result["face"]["landmarks_2d_ref"]["nan_policy"] == "preserve"
    assert result["face"]["landmarks_2d_ref"]["interpolation_policy"] == "none"
    assert result["face"]["landmarks_3d_ref"] is None

    assert result["hands"]["left"]["enabled"] is True
    assert result["hands"]["left"]["landmark_count"] == HAND_LANDMARK_COUNT
    assert result["hands"]["right"]["enabled"] is False
    assert result["hands"]["right"]["landmark_count"] == 0
    assert result["hands"]["right"]["present_ref"] is None
    assert result["hands"]["right"]["bbox_ref"] is None
    assert result["hands"]["right"]["landmarks_2d_ref"] is None
    assert result["hands"]["right"]["handedness_ref"] is None

    uri = result["face"]["landmarks_2d_ref"]["uri"]
    assert uri in refinement_sidecars
    assert os.path.isfile(refinement_sidecars[uri])
    with np.load(refinement_sidecars[uri], allow_pickle=False) as data:
        assert data["face_landmarks_2d"].shape == (3, FACE_LANDMARK_COUNT, 2)
        assert np.all(np.isfinite(data["face_landmarks_2d"]))
        assert data["left_hand_landmarks_2d"].shape == (3, HAND_LANDMARK_COUNT, 2)
        assert np.all(np.isnan(data["left_hand_landmarks_2d"][0]))
        assert np.all(np.isfinite(data["left_hand_landmarks_2d"][1]))
        assert np.all(np.isnan(data["left_hand_landmarks_2d"][2]))
        assert np.all(np.isnan(data["right_hand_landmarks_2d"]))
        assert data["left_hand_present"].tolist() == pytest.approx([0.0, 0.82, 0.0])
        assert data["right_hand_present"].tolist() == pytest.approx([0.0, 0.0, 0.0])

    report = refinement_sidecars[report_ref["uri"]]
    assert report["identity_inference_performed"] is False
    assert report["biometric_embedding_exported"] is False
    assert report["characters"][0]["face_frames"] == 3
    assert report["characters"][0]["left_hand_frames"] == 1
    assert report["characters"][0]["right_hand_frames"] == 0


def test_completely_absent_face_and_hands_emit_disabled_null_refs(three_frame_video, tmp_path):
    character, sidecars = _tracking_character(tmp_path)
    characters, _, _ = run_face_hand_refinement(
        str(three_frame_video),
        characters=[character],
        frame_count=3,
        refiner=_NoObservationRefiner(),
        output_dir=str(tmp_path / "out"),
        sidecars=sidecars,
    )
    result = characters[0]
    assert result["face"]["enabled"] is False
    assert result["face"]["landmark_count"] == 0
    assert result["face"]["bbox_ref"] is None
    assert result["face"]["landmarks_2d_ref"] is None
    for side in ("left", "right"):
        assert result["hands"][side]["enabled"] is False
        assert result["hands"][side]["landmark_count"] == 0
        assert result["hands"][side]["present_ref"] is None
        assert result["hands"][side]["bbox_ref"] is None
        assert result["hands"][side]["landmarks_2d_ref"] is None
        assert result["hands"][side]["handedness_ref"] is None


def test_refinement_rejects_wrong_face_landmark_shape(three_frame_video, tmp_path):
    character, sidecars = _tracking_character(tmp_path)
    with pytest.raises(FaceHandRefinementError, match="Face landmarks must have shape"):
        run_face_hand_refinement(
            str(three_frame_video),
            characters=[character],
            frame_count=3,
            refiner=_InvalidFaceRefiner(),
            output_dir=str(tmp_path / "out"),
            sidecars=sidecars,
        )
