import pytest

from packages.pipeline_core.mediapipe_face_hand_backend import _face_search_bbox


def test_tall_full_body_bbox_uses_shorter_upper_body_face_roi() -> None:
    face_bbox = _face_search_bbox((20.0, 10.0, 140.0, 410.0))

    assert face_bbox == pytest.approx((20.0, 10.0, 140.0, 160.0))


def test_near_square_person_bbox_keeps_complete_roi() -> None:
    face_bbox = _face_search_bbox((20.0, 10.0, 140.0, 130.0))

    assert face_bbox == pytest.approx((20.0, 10.0, 140.0, 130.0))


def test_face_roi_preserves_full_person_width() -> None:
    x1, _, x2, _ = _face_search_bbox((32.0, 18.0, 212.0, 558.0))

    assert (x1, x2) == pytest.approx((32.0, 212.0))
