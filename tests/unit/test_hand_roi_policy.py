import pytest

from packages.pipeline_core.mediapipe_face_hand_backend import _hand_search_bbox


def test_tall_full_body_bbox_removes_lower_leg_region_for_hand_search() -> None:
    hand_bbox = _hand_search_bbox((20.0, 10.0, 140.0, 410.0))

    assert hand_bbox == pytest.approx((20.0, 10.0, 140.0, 298.0))


def test_near_square_person_bbox_keeps_complete_hand_roi() -> None:
    hand_bbox = _hand_search_bbox((20.0, 10.0, 140.0, 130.0))

    assert hand_bbox == pytest.approx((20.0, 10.0, 140.0, 130.0))


def test_hand_roi_preserves_complete_person_width() -> None:
    x1, _, x2, _ = _hand_search_bbox((32.0, 18.0, 212.0, 558.0))

    assert (x1, x2) == pytest.approx((32.0, 212.0))
