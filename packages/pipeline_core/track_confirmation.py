from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import isfinite
from statistics import fmean


class TrackConfirmationError(ValueError):
    """Raised when track-confirmation evidence or policy is invalid."""


@dataclass(frozen=True)
class TrackConfirmationPolicy:
    """Evidence policy applied after low-threshold candidate association.

    The detector may run with a permissive candidate threshold (for example 0.30)
    to preserve temporal recall. This policy decides whether an associated track has
    enough aggregate evidence to enter the canonical character set.
    """

    strong_single_detection_threshold: float = 0.70
    repeated_top_k_mean_threshold: float = 0.40
    repeated_min_detections: int = 2
    repeated_top_k: int = 3
    sustained_mean_threshold: float = 0.35
    sustained_min_detections: int = 5

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class TrackConfirmationEvidence:
    candidate_character_id: str
    confirmed: bool
    reason: str
    detection_count: int
    max_detection_score: float
    mean_detection_score: float
    top_k_mean_detection_score: float

    def to_dict(self) -> dict[str, str | bool | int | float]:
        return asdict(self)


def validate_track_confirmation_policy(policy: TrackConfirmationPolicy) -> None:
    thresholds = {
        "strong_single_detection_threshold": policy.strong_single_detection_threshold,
        "repeated_top_k_mean_threshold": policy.repeated_top_k_mean_threshold,
        "sustained_mean_threshold": policy.sustained_mean_threshold,
    }
    for name, value in thresholds.items():
        if not isfinite(value) or not 0.0 <= value <= 1.0:
            raise TrackConfirmationError(f"{name} must be finite and within [0, 1]")
    if policy.repeated_min_detections < 1:
        raise TrackConfirmationError("repeated_min_detections must be >= 1")
    if policy.repeated_top_k < 1:
        raise TrackConfirmationError("repeated_top_k must be >= 1")
    if policy.sustained_min_detections < 1:
        raise TrackConfirmationError("sustained_min_detections must be >= 1")


def evaluate_track_scores(
    candidate_character_id: str,
    scores: Sequence[float],
    *,
    policy: TrackConfirmationPolicy | None = None,
) -> TrackConfirmationEvidence:
    """Confirm one candidate track from aggregate detector evidence.

    Confirmation is deliberately track-level rather than a higher frame-level cutoff:
    a candidate can pass via one very strong anchor, repeated moderate evidence, or
    sustained evidence whose mean clears the detector's normal confidence regime.
    Weak candidates are rejected but remain visible in diagnostics.
    """

    cfg = policy or TrackConfirmationPolicy()
    validate_track_confirmation_policy(cfg)
    if not scores:
        raise TrackConfirmationError("track confirmation requires at least one detection score")

    normalized_scores: list[float] = []
    for score in scores:
        value = float(score)
        if not isfinite(value) or not 0.0 <= value <= 1.0:
            raise TrackConfirmationError("detection scores must be finite and within [0, 1]")
        normalized_scores.append(value)

    ordered = sorted(normalized_scores, reverse=True)
    top_k_scores = ordered[: min(cfg.repeated_top_k, len(ordered))]
    max_score = ordered[0]
    mean_score = fmean(normalized_scores)
    top_k_mean = fmean(top_k_scores)
    count = len(normalized_scores)

    if max_score >= cfg.strong_single_detection_threshold:
        confirmed = True
        reason = "strong_single_detection"
    elif (
        count >= cfg.repeated_min_detections
        and top_k_mean >= cfg.repeated_top_k_mean_threshold
    ):
        confirmed = True
        reason = "repeated_moderate_evidence"
    elif count >= cfg.sustained_min_detections and mean_score >= cfg.sustained_mean_threshold:
        confirmed = True
        reason = "sustained_track_evidence"
    else:
        confirmed = False
        reason = "insufficient_track_evidence"

    return TrackConfirmationEvidence(
        candidate_character_id=candidate_character_id,
        confirmed=confirmed,
        reason=reason,
        detection_count=count,
        max_detection_score=round(max_score, 6),
        mean_detection_score=round(mean_score, 6),
        top_k_mean_detection_score=round(top_k_mean, 6),
    )
