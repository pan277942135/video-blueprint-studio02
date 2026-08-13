from types import SimpleNamespace

import numpy as np
import pytest

from packages.pipeline_core import mediapipe_face_hand_backend as backend


class _FakeTask:
    def __init__(self, kind: str):
        self.kind = kind
        self.timestamps: list[int] = []
        self.closed = False

    def detect_for_video(self, image, timestamp_ms: int):
        self.timestamps.append(timestamp_ms)
        if self.kind == "face":
            landmarks = [
                SimpleNamespace(x=0.2 + (index % 10) * 0.01, y=0.2 + (index % 12) * 0.01, presence=0.9)
                for index in range(478)
            ]
            return SimpleNamespace(face_landmarks=[landmarks])
        landmarks = [
            SimpleNamespace(x=0.55 + (index % 5) * 0.02, y=0.45 + (index % 7) * 0.02, presence=0.8)
            for index in range(21)
        ]
        handedness = [[SimpleNamespace(category_name="Left", score=0.93)]]
        return SimpleNamespace(hand_landmarks=[landmarks], handedness=handedness)

    def close(self):
        self.closed = True


class _FakeFactory:
    def __init__(self, kind: str):
        self.kind = kind
        self.instances: list[_FakeTask] = []

    def create_from_options(self, options):
        instance = _FakeTask(self.kind)
        self.instances.append(instance)
        return instance


class _FakeOptions:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeImage:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _fake_mediapipe():
    face_factory = _FakeFactory("face")
    hand_factory = _FakeFactory("hand")
    vision = SimpleNamespace(
        RunningMode=SimpleNamespace(VIDEO="VIDEO"),
        FaceLandmarkerOptions=_FakeOptions,
        HandLandmarkerOptions=_FakeOptions,
        FaceLandmarker=face_factory,
        HandLandmarker=hand_factory,
    )
    module = SimpleNamespace(
        __version__="0.10.35",
        tasks=SimpleNamespace(vision=vision, BaseOptions=_FakeOptions),
        Image=_FakeImage,
        ImageFormat=SimpleNamespace(SRGB="SRGB"),
    )
    return module, face_factory, hand_factory


def _config(tmp_path):
    face = tmp_path / "face.task"
    hand = tmp_path / "hand.task"
    face.write_bytes(b"face")
    hand.write_bytes(b"hand")
    return backend.MediaPipeFaceHandConfig(
        face_task_path=str(face),
        hand_task_path=str(hand),
        fps_num=30,
        fps_den=1,
    )


def test_backend_requires_explicit_approval(monkeypatch, tmp_path):
    config = _config(tmp_path)
    monkeypatch.setattr(
        backend,
        "_sha256_file",
        lambda path: backend.APPROVED_FACE_TASK_SHA256 if path.endswith("face.task") else backend.APPROVED_HAND_TASK_SHA256,
    )
    fake_mp, _, _ = _fake_mediapipe()
    with pytest.raises(backend.MediaPipeFaceHandConfigurationError, match="explicit project-owner approval"):
        backend.MediaPipeFaceHandRefiner(config, artifacts_approved=False, mediapipe_module=fake_mp)


def test_backend_rejects_unapproved_artifact_hash(monkeypatch, tmp_path):
    config = _config(tmp_path)
    monkeypatch.setattr(backend, "_sha256_file", lambda path: "0" * 64)
    fake_mp, _, _ = _fake_mediapipe()
    with pytest.raises(backend.MediaPipeFaceHandConfigurationError, match="FaceLandmarker task SHA256 mismatch"):
        backend.MediaPipeFaceHandRefiner(config, artifacts_approved=True, mediapipe_module=fake_mp)


def test_backend_maps_face_and_hand_to_full_frame_and_uses_monotonic_video_timestamps(monkeypatch, tmp_path):
    config = _config(tmp_path)
    monkeypatch.setattr(
        backend,
        "_sha256_file",
        lambda path: backend.APPROVED_FACE_TASK_SHA256 if path.endswith("face.task") else backend.APPROVED_HAND_TASK_SHA256,
    )
    fake_mp, face_factory, hand_factory = _fake_mediapipe()
    refiner = backend.MediaPipeFaceHandRefiner(config, artifacts_approved=True, mediapipe_module=fake_mp)

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    first = refiner.refine(frame, (20.0, 10.0, 140.0, 110.0), 0, "char_000")
    second = refiner.refine(frame, (20.0, 10.0, 140.0, 110.0), 1, "char_000")

    assert first is not None and second is not None
    assert first.face is not None
    assert first.face.landmarks_xy.shape == (478, 2)
    assert len(first.hands) == 1
    assert first.hands[0].side == "left"
    assert first.hands[0].landmarks_xy.shape == (21, 2)
    assert first.hands[0].handedness_confidence == pytest.approx(0.93)
    assert np.all(first.face.landmarks_xy[:, 0] >= 0.0)
    assert np.all(first.face.landmarks_xy[:, 0] < 160.0)
    assert np.all(first.face.landmarks_xy[:, 1] >= 0.0)
    assert np.all(first.face.landmarks_xy[:, 1] < 120.0)

    assert len(face_factory.instances) == 1
    assert len(hand_factory.instances) == 1
    assert face_factory.instances[0].timestamps == [0, 33]
    assert hand_factory.instances[0].timestamps == [0, 33]
    refiner.close()
    assert face_factory.instances[0].closed is True
    assert hand_factory.instances[0].closed is True


def test_from_environment_is_disabled_only_when_completely_unconfigured(monkeypatch):
    for key in ("VBS_MEDIAPIPE_FACE_TASK", "VBS_MEDIAPIPE_HAND_TASK", "VBS_MEDIAPIPE_TASKS_APPROVED"):
        monkeypatch.delenv(key, raising=False)
    assert backend.MediaPipeFaceHandRefiner.from_environment(fps_num=30, fps_den=1) is None

    monkeypatch.setenv("VBS_MEDIAPIPE_FACE_TASK", "/tmp/face.task")
    with pytest.raises(backend.MediaPipeFaceHandConfigurationError, match="Incomplete E3.2 MediaPipe configuration"):
        backend.MediaPipeFaceHandRefiner.from_environment(fps_num=30, fps_den=1)
