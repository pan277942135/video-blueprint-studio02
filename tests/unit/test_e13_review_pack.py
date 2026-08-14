from packages.pipeline_core.e13_review_pack import ReviewFrame, select_review_frames


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


def test_select_review_frames_falls_back_to_first_middle_last() -> None:
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
