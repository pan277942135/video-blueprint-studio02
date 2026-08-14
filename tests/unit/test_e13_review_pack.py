import hashlib
import json
import zipfile

from packages.pipeline_core.e13_review_pack import (
    ReviewFrame,
    _normalized_analysis_media,
    select_review_frames,
)


def test_select_review_frames_uses_shot_keyframes_and_deduplicates() -> None:
    blueprint = {
        "shots": [
            {
                "shot_id": "shot_000",
                "keyframes": [
                    {"frame_idx": 0, "time_us": 0, "kind": "first"},
                    {"frame_idx": 10, "time_us": 1_000_000, "kind": "middle"},
                ],
            },
            {
                "shot_id": "shot_001",
                "keyframes": [
                    {"frame_idx": 10, "time_us": 1_000_000, "kind": "first"},
                    {"frame_idx": 20, "time_us": 2_000_000, "kind": "last"},
                ],
            },
        ]
    }

    assert select_review_frames(blueprint) == [
        ReviewFrame(frame_idx=0, time_us=0, shot_id="shot_000", kind="first"),
        ReviewFrame(frame_idx=10, time_us=1_000_000, shot_id="shot_000", kind="middle"),
        ReviewFrame(frame_idx=20, time_us=2_000_000, shot_id="shot_001", kind="last"),
    ]


def test_select_review_frames_prefers_normalized_timebase_for_fallback() -> None:
    blueprint = {
        "timebase": {
            "frame_count": 11,
            "frame_duration_us": 100_000.0,
        },
        "source_video": {
            "source_frame_count": 21,
            "duration_us": 1_000_000,
        },
        "shots": [],
    }

    assert select_review_frames(blueprint) == [
        ReviewFrame(frame_idx=0, time_us=0, shot_id=None, kind="fallback"),
        ReviewFrame(frame_idx=5, time_us=500_000, shot_id=None, kind="fallback"),
        ReviewFrame(frame_idx=10, time_us=1_000_000, shot_id=None, kind="fallback"),
    ]


def test_select_review_frames_falls_back_to_source_when_timebase_missing() -> None:
    blueprint = {
        "source_video": {
            "source_frame_count": 11,
            "duration_us": 1_000_000,
        },
        "shots": [],
    }

    assert select_review_frames(blueprint) == [
        ReviewFrame(frame_idx=0, time_us=0, shot_id=None, kind="fallback"),
        ReviewFrame(frame_idx=5, time_us=500_000, shot_id=None, kind="fallback"),
        ReviewFrame(frame_idx=10, time_us=1_000_000, shot_id=None, kind="fallback"),
    ]


def test_select_review_frames_caps_evenly() -> None:
    blueprint = {
        "shots": [
            {
                "shot_id": "shot_000",
                "keyframes": [
                    {"frame_idx": idx, "time_us": idx * 100_000, "kind": "sample"}
                    for idx in range(10)
                ],
            }
        ]
    }

    selected = select_review_frames(blueprint, max_frames=4)

    assert [item.frame_idx for item in selected] == [0, 3, 6, 9]


def test_normalized_analysis_media_is_manifest_pinned(tmp_path) -> None:
    payload = b"normalized-analysis-video"
    digest = hashlib.sha256(payload).hexdigest()
    bundle_path = tmp_path / "bundle.zip"
    manifest = {
        "files": [
            {
                "path": "artifacts/normalized/analysis_cfr.mp4",
                "size": len(payload),
                "sha256": digest,
            }
        ]
    }

    with zipfile.ZipFile(bundle_path, "w") as archive:
        archive.writestr("bundle_manifest.json", json.dumps(manifest))
        archive.writestr("artifacts/normalized/analysis_cfr.mp4", payload)

    with zipfile.ZipFile(bundle_path, "r") as archive:
        uri, actual_payload, actual_sha256 = _normalized_analysis_media(archive)

    assert uri == "artifacts/normalized/analysis_cfr.mp4"
    assert actual_payload == payload
    assert actual_sha256 == digest
