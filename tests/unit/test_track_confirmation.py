import pytest

from packages.pipeline_core.track_confirmation import (
    TrackConfirmationError,
    TrackConfirmationPolicy,
    evaluate_track_scores,
    validate_track_confirmation_policy,
)


def test_rejects_persistent_weak_candidate_without_raising_candidate_floor() -> None:
    evidence = evaluate_track_scores(
        "char_001",
        [0.31, 0.32, 0.34, 0.35, 0.39, 0.33],
    )

    assert evidence.confirmed is False
    assert evidence.reason == "insufficient_track_evidence"
    assert evidence.detection_count == 6
    assert evidence.max_detection_score == 0.39
    assert evidence.mean_detection_score == pytest.approx(0.34)
    assert evidence.top_k_mean_detection_score == pytest.approx(0.36)


def test_accepts_repeated_moderate_candidate() -> None:
    evidence = evaluate_track_scores("char_000", [0.41, 0.42])

    assert evidence.confirmed is True
    assert evidence.reason == "repeated_moderate_evidence"


def test_accepts_single_strong_anchor() -> None:
    evidence = evaluate_track_scores("char_000", [0.82])

    assert evidence.confirmed is True
    assert evidence.reason == "strong_single_detection"


def test_accepts_sustained_track_above_normal_confidence_regime() -> None:
    evidence = evaluate_track_scores(
        "char_000",
        [0.35, 0.36, 0.35, 0.36, 0.35],
    )

    assert evidence.confirmed is True
    assert evidence.reason == "sustained_track_evidence"


def test_policy_validation_rejects_invalid_thresholds() -> None:
    with pytest.raises(TrackConfirmationError):
        validate_track_confirmation_policy(
            TrackConfirmationPolicy(repeated_top_k_mean_threshold=1.01)
        )


def test_score_validation_rejects_non_finite_values() -> None:
    with pytest.raises(TrackConfirmationError):
        evaluate_track_scores("char_000", [float("nan")])
