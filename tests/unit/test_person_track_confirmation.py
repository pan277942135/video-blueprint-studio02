from packages.pipeline_core.person_tracking import (
    AnonymousTrack,
    PersonDetection,
    confirm_anonymous_tracks,
)


def _track(character_id: str, scores: list[float]) -> AnonymousTrack:
    detections = {
        frame_idx: PersonDetection(
            frame_idx=frame_idx,
            bbox_xyxy=(10.0, 10.0, 100.0, 200.0),
            score=score,
        )
        for frame_idx, score in enumerate(scores)
    }
    return AnonymousTrack(
        character_id=character_id,
        track_label=f"anonymous_track_{character_id.removeprefix('char_')}",
        shot_id="shot_000",
        detections=detections,
    )


def test_confirmation_rejects_weak_middle_candidate_and_reindexes_ids() -> None:
    tracks = [
        _track("char_000", [0.91]),
        _track("char_001", [0.31, 0.33, 0.35, 0.36, 0.34]),
        _track("char_002", [0.82]),
    ]

    confirmed, evidence = confirm_anonymous_tracks(tracks)

    assert [track.character_id for track in confirmed] == ["char_000", "char_001"]
    assert [track.track_label for track in confirmed] == [
        "anonymous_track_000",
        "anonymous_track_001",
    ]
    assert evidence[0]["candidate_character_id"] == "char_000"
    assert evidence[0]["confirmed_character_id"] == "char_000"
    assert evidence[1]["candidate_character_id"] == "char_001"
    assert evidence[1]["confirmed"] is False
    assert evidence[1]["confirmed_character_id"] is None
    assert evidence[2]["candidate_character_id"] == "char_002"
    assert evidence[2]["confirmed_character_id"] == "char_001"
