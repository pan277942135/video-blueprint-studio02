from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from packages.pipeline_core.person_mask import decode_binary_rle, encode_binary_rle
from packages.pipeline_core.sparse_motion import atomic_npz, sha256_file


class EnvironmentPhotometryError(RuntimeError):
    pass


@dataclass(frozen=True)
class EnvironmentPhotometryConfig:
    min_background_fraction: float = 0.10
    mask_erosion_radius_px: int = 2
    exposure_epsilon: float = 1e-4

    def validate(self) -> None:
        if not 0.0 < self.min_background_fraction <= 1.0:
            raise EnvironmentPhotometryError("min_background_fraction must be within (0, 1]")
        if self.mask_erosion_radius_px < 0:
            raise EnvironmentPhotometryError("mask_erosion_radius_px must be >= 0")
        if self.exposure_epsilon <= 0.0:
            raise EnvironmentPhotometryError("exposure_epsilon must be positive")

    def token(self) -> str:
        return hashlib.sha256(json.dumps(self.__dict__, sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_environment(cls) -> EnvironmentPhotometryConfig | None:
        raw = os.environ.get("VBS_E9_ENVIRONMENT_ENABLED")
        if raw is None or raw.strip().lower() == "false":
            return None
        if raw.strip().lower() != "true":
            raise EnvironmentPhotometryError("VBS_E9_ENVIRONMENT_ENABLED must be 'true' or 'false'")
        value = cls()
        value.validate()
        return value


def _atomic_json(path: str, payload: dict[str, Any]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"), sort_keys=True)
    os.replace(tmp, path)


def _presence_mask(character: dict[str, Any], frame_count: int) -> np.ndarray:
    result = np.zeros(frame_count, dtype=np.bool_)
    presence = character.get("presence")
    if not isinstance(presence, list):
        raise EnvironmentPhotometryError("E9 character presence must be an array")
    for interval in presence:
        if not isinstance(interval, dict):
            raise EnvironmentPhotometryError("E9 character presence interval must be an object")
        frame_start = interval.get("frame_start")
        frame_end = interval.get("frame_end")
        if not isinstance(frame_start, int) or not isinstance(frame_end, int):
            raise EnvironmentPhotometryError("E9 character presence frame bounds must be integers")
        if frame_start < 0 or frame_end < frame_start or frame_end >= frame_count:
            raise EnvironmentPhotometryError("E9 character presence frame bounds are invalid")
        result[frame_start : frame_end + 1] = True
    return result


def _load_character_masks(
    character: dict[str, Any],
    *,
    sidecars: dict[str, Any],
    frame_count: int,
    height: int,
    width: int,
) -> tuple[np.ndarray, list[np.ndarray | None]]:
    presence = _presence_mask(character, frame_count)
    ref = character.get("person_mask_ref")
    if not isinstance(ref, dict):
        raise EnvironmentPhotometryError("E9 requires E4 person_mask_ref for every anonymous character")
    uri = ref.get("uri")
    if not isinstance(uri, str):
        raise EnvironmentPhotometryError("E9 person mask URI is missing")
    path_value = sidecars.get(uri)
    if not isinstance(path_value, (str, os.PathLike)) or not os.path.isfile(str(path_value)):
        raise EnvironmentPhotometryError(f"E9 physical person mask unavailable: {uri}")
    path = str(path_value)
    if ref.get("checksum_sha256") != sha256_file(path):
        raise EnvironmentPhotometryError(f"E9 person mask checksum mismatch: {uri}")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("encoding") != "row_major_binary_rle_v1":
        raise EnvironmentPhotometryError("E9 only accepts certified row_major_binary_rle_v1 person masks")
    if payload.get("frame_count") != frame_count or payload.get("height") != height or payload.get("width") != width:
        raise EnvironmentPhotometryError("E9 person mask geometry does not match normalized video")
    frames = payload.get("frames")
    if not isinstance(frames, list) or len(frames) != frame_count:
        raise EnvironmentPhotometryError("E9 person mask frame count mismatch")
    decoded: list[np.ndarray | None] = []
    for frame_idx, encoded in enumerate(frames):
        if encoded is None:
            decoded.append(None)
            continue
        mask = decode_binary_rle(encoded)
        if mask.shape != (height, width):
            raise EnvironmentPhotometryError(f"E9 decoded person mask shape mismatch at frame {frame_idx}")
        decoded.append(mask.astype(np.bool_, copy=False))
    return presence, decoded


def _shot_start_mask(shots: list[dict[str, Any]], frame_count: int) -> np.ndarray:
    result = np.zeros(frame_count, dtype=np.bool_)
    for shot in shots:
        if not isinstance(shot, dict):
            continue
        frame_start = shot.get("frame_start")
        frame_end = shot.get("frame_end")
        if not isinstance(frame_start, int) or not isinstance(frame_end, int):
            raise EnvironmentPhotometryError("E9 shot bounds must be integers")
        if frame_start < 0 or frame_end < frame_start or frame_end >= frame_count:
            raise EnvironmentPhotometryError("E9 shot bounds are invalid")
        result[frame_start] = True
    return result


def _background_metrics(
    frame_bgr: np.ndarray,
    background_mask: np.ndarray,
    *,
    erosion_radius_px: int,
) -> tuple[float, np.ndarray, float]:
    frame = frame_bgr.astype(np.float32) / 255.0
    blue = frame[:, :, 0]
    green = frame[:, :, 1]
    red = frame[:, :, 2]
    luma = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    luminance = float(np.median(luma[background_mask]))

    channel_sum = red + green + blue
    chroma_valid = background_mask & (channel_sum > 1e-6)
    if int(np.count_nonzero(chroma_valid)) < 16:
        chroma = np.asarray([math.nan, math.nan], dtype=np.float32)
    else:
        r_chroma = red[chroma_valid] / channel_sum[chroma_valid]
        b_chroma = blue[chroma_valid] / channel_sum[chroma_valid]
        chroma = np.asarray([np.median(r_chroma), np.median(b_chroma)], dtype=np.float32)

    sample_mask: np.ndarray = background_mask.astype(np.uint8) * 255
    if erosion_radius_px > 0:
        size = erosion_radius_px * 2 + 1
        kernel = np.ones((size, size), dtype=np.uint8)
        sample_mask = cv2.erode(sample_mask, kernel, iterations=1)
    sample = sample_mask > 0
    if int(np.count_nonzero(sample)) < 16:
        sharpness = math.nan
    else:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        values = laplacian[sample]
        sharpness = float(np.var(values, dtype=np.float64))
    return luminance, chroma, sharpness


def _series_ref(
    *,
    uri: str,
    checksum: str,
    frame_count: int,
    shape: list[int],
    axes: list[str],
    unit: str,
    array_key: str,
    valid_array_key: str,
    config: EnvironmentPhotometryConfig,
    extra_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "uri": uri,
        "format": "npz",
        "dtype": "float32",
        "shape": shape,
        "axes": axes,
        "unit": unit,
        "coordinate_space": "none",
        "sampling": "per_frame",
        "frame_start": 0,
        "frame_end": frame_count - 1,
        "compression": "zip",
        "nan_policy": "preserve",
        "interpolation_policy": "none",
        "checksum_sha256": checksum,
        "metadata": {
            "array_key": array_key,
            "valid_array_key": valid_array_key,
            "algorithm": "background_photometry_v1",
            "background_definition": "inverse_union_of_e4_person_masks",
            "requires_complete_masks_for_present_tracks": True,
            "config_sha256": config.token(),
            **extra_metadata,
        },
    }


def run_environment_photometry(
    video_path: str,
    *,
    blueprint: dict[str, Any],
    output_dir: str,
    sidecars: dict[str, Any],
    config: EnvironmentPhotometryConfig,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config.validate()
    if not os.path.isfile(video_path):
        raise EnvironmentPhotometryError(f"E9 normalized video missing: {video_path}")
    frame_count = int(blueprint["timebase"]["frame_count"])
    width = int(blueprint["source_video"]["width"])
    height = int(blueprint["source_video"]["height"])
    characters = blueprint.get("characters")
    if not isinstance(characters, list):
        raise EnvironmentPhotometryError("E9 requires characters array")

    character_masks = [
        _load_character_masks(
            character,
            sidecars=sidecars,
            frame_count=frame_count,
            height=height,
            width=width,
        )
        for character in characters
        if isinstance(character, dict)
    ]
    if len(character_masks) != len(characters):
        raise EnvironmentPhotometryError("E9 character manifest contains non-object rows")

    luminance = np.full(frame_count, np.nan, dtype=np.float32)
    exposure_change = np.full(frame_count, np.nan, dtype=np.float32)
    white_balance_proxy = np.full((frame_count, 2), np.nan, dtype=np.float32)
    blur_proxy = np.full(frame_count, np.nan, dtype=np.float32)
    valid_frame = np.zeros(frame_count, dtype=np.bool_)
    exposure_valid = np.zeros(frame_count, dtype=np.bool_)
    background_fraction = np.full(frame_count, np.nan, dtype=np.float32)
    mask_frames: list[dict[str, Any] | None] = [None] * frame_count
    shot_start = _shot_start_mask(blueprint.get("shots", []), frame_count)

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise EnvironmentPhotometryError("OpenCV could not open normalized video for E9")
    decoded = 0
    try:
        while decoded < frame_count:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if frame.shape[:2] != (height, width):
                raise EnvironmentPhotometryError(f"E9 decoded frame {decoded} geometry mismatch")
            union_person = np.zeros((height, width), dtype=np.bool_)
            evidence_complete = True
            for presence, masks in character_masks:
                if not bool(presence[decoded]):
                    continue
                mask = masks[decoded]
                if mask is None:
                    evidence_complete = False
                    break
                union_person |= mask
            if evidence_complete:
                background = ~union_person
                fraction = float(np.count_nonzero(background)) / float(height * width)
                background_fraction[decoded] = fraction
                if fraction >= config.min_background_fraction:
                    y_value, chroma, sharpness = _background_metrics(
                        frame,
                        background,
                        erosion_radius_px=config.mask_erosion_radius_px,
                    )
                    luminance[decoded] = y_value
                    white_balance_proxy[decoded] = chroma
                    blur_proxy[decoded] = sharpness
                    valid_frame[decoded] = bool(
                        math.isfinite(y_value)
                        and np.all(np.isfinite(chroma))
                        and math.isfinite(sharpness)
                    )
                    if valid_frame[decoded]:
                        mask_frames[decoded] = encode_binary_rle(background)
            decoded += 1
    finally:
        capture.release()
    if decoded != frame_count:
        raise EnvironmentPhotometryError(f"E9 decoded {decoded}/{frame_count} normalized frames")

    for frame_idx in range(1, frame_count):
        if shot_start[frame_idx] or not valid_frame[frame_idx] or not valid_frame[frame_idx - 1]:
            continue
        previous = max(float(luminance[frame_idx - 1]), config.exposure_epsilon)
        current = max(float(luminance[frame_idx]), config.exposure_epsilon)
        exposure_change[frame_idx] = math.log2(current / previous)
        exposure_valid[frame_idx] = True

    mask_uri = "artifacts/timeseries/environment_background_mask.rle.json"
    mask_path = os.path.join(output_dir, mask_uri.replace("/", os.sep))
    Path(mask_path).parent.mkdir(parents=True, exist_ok=True)
    mask_payload = {
        "encoding": "row_major_binary_rle_v1",
        "frame_count": frame_count,
        "height": height,
        "width": width,
        "frames": mask_frames,
    }
    _atomic_json(mask_path, mask_payload)
    mask_checksum = sha256_file(mask_path)
    background_mask_ref = {
        "uri": mask_uri,
        "format": "rle_json",
        "dtype": "uint8",
        "shape": [frame_count, height, width],
        "axes": ["frame", "y", "x"],
        "unit": "binary",
        "coordinate_space": "pixel_xy",
        "sampling": "per_frame",
        "frame_start": 0,
        "frame_end": frame_count - 1,
        "compression": "none",
        "nan_policy": "preserve",
        "interpolation_policy": "none",
        "checksum_sha256": mask_checksum,
        "metadata": {
            "encoding": "row_major_binary_rle_v1",
            "foreground_semantics": "background_pixel",
            "background_definition": "inverse_union_of_e4_person_masks",
            "requires_complete_masks_for_present_tracks": True,
            "invalid_frame_value": "null",
            "validity_array_uri": "artifacts/timeseries/environment_photometry.npz",
            "validity_array_key": "valid_frame",
            "config_sha256": config.token(),
        },
    }

    metrics_uri = "artifacts/timeseries/environment_photometry.npz"
    metrics_path = os.path.join(output_dir, metrics_uri.replace("/", os.sep))
    atomic_npz(
        metrics_path,
        luminance=luminance,
        exposure_change_ev_proxy=exposure_change,
        white_balance_chromaticity_rb=white_balance_proxy,
        blur_laplacian_variance=blur_proxy,
        valid_frame=valid_frame,
        exposure_valid=exposure_valid,
        background_fraction=background_fraction,
    )
    metrics_checksum = sha256_file(metrics_path)
    luminance_ref = _series_ref(
        uri=metrics_uri,
        checksum=metrics_checksum,
        frame_count=frame_count,
        shape=[frame_count],
        axes=["frame"],
        unit="normalized_bt709_luma",
        array_key="luminance",
        valid_array_key="valid_frame",
        config=config,
        extra_metadata={"statistic": "median_background_luma", "range": [0.0, 1.0]},
    )
    exposure_ref = _series_ref(
        uri=metrics_uri,
        checksum=metrics_checksum,
        frame_count=frame_count,
        shape=[frame_count],
        axes=["frame"],
        unit="ev_proxy",
        array_key="exposure_change_ev_proxy",
        valid_array_key="exposure_valid",
        config=config,
        extra_metadata={
            "definition": "log2(current_background_median_luma/previous_background_median_luma)",
            "shot_boundary_reset": True,
            "not_camera_exif_exposure": True,
        },
    )
    white_balance_ref = _series_ref(
        uri=metrics_uri,
        checksum=metrics_checksum,
        frame_count=frame_count,
        shape=[frame_count, 2],
        axes=["frame", "chromaticity_component"],
        unit="chromaticity_ratio",
        array_key="white_balance_chromaticity_rb",
        valid_array_key="valid_frame",
        config=config,
        extra_metadata={
            "components": ["median_r_over_rgb_sum", "median_b_over_rgb_sum"],
            "proxy_only": True,
            "not_camera_white_balance_metadata": True,
        },
    )
    blur_ref = _series_ref(
        uri=metrics_uri,
        checksum=metrics_checksum,
        frame_count=frame_count,
        shape=[frame_count],
        axes=["frame"],
        unit="normalized_luma_laplacian_variance",
        array_key="blur_laplacian_variance",
        valid_array_key="valid_frame",
        config=config,
        extra_metadata={
            "measure_semantics": "background_sharpness_proxy_higher_is_sharper",
            "not_physical_depth_of_field_or_psf": True,
            "mask_erosion_radius_px": config.mask_erosion_radius_px,
        },
    )

    valid_count = int(np.count_nonzero(valid_frame))
    exposure_count = int(np.count_nonzero(exposure_valid))
    coverage = valid_count / frame_count if frame_count else 0.0
    finite_fraction = background_fraction[np.isfinite(background_fraction)]
    report_uri = "artifacts/reports/environment_photometry.json"
    report = {
        "stage": "environment",
        "algorithm": "background_photometry_v1",
        "frame_count": frame_count,
        "valid_frame_count": valid_count,
        "exposure_change_valid_count": exposure_count,
        "coverage": round(coverage, 6),
        "background_fraction_p50": round(float(np.median(finite_fraction)), 6) if len(finite_fraction) else None,
        "depth_emitted": False,
        "occluder_tracks_emitted": False,
        "semantic_scene_inference_performed": False,
        "weather_inference_performed": False,
        "material_inference_performed": False,
        "identity_inference_performed": False,
        "biometric_embedding_exported": False,
        "new_model_weights_introduced": False,
    }
    report_path = os.path.join(output_dir, report_uri.replace("/", os.sep))
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(report_path, report)
    report_ref = {"kind": "environment_photometry", "uri": report_uri, "sha256": sha256_file(report_path)}
    extension = {
        "enabled": True,
        "algorithm": "background_photometry_v1",
        "background_definition": "inverse_union_of_e4_person_masks",
        "requires_complete_masks_for_present_tracks": True,
        "depth_emitted": False,
        "occluder_tracks_emitted": False,
        "semantic_scene_inference_performed": False,
        "weather_inference_performed": False,
        "material_inference_performed": False,
        "identity_inference_performed": False,
        "biometric_embedding_exported": False,
        "new_model_weights_introduced": False,
        "valid_frame_count": valid_count,
        "coverage": round(coverage, 6),
        "min_background_fraction": config.min_background_fraction,
        "mask_erosion_radius_px": config.mask_erosion_radius_px,
        "config_sha256": config.token(),
    }
    quality = {
        "score": round(coverage, 6),
        "coverage": round(coverage, 6),
        "warnings": [
            "E9.1 is photometric evidence only; depth, semantic scene, weather and material inference are not performed",
            "blur_ref stores a background sharpness proxy (Laplacian variance), not a calibrated physical blur kernel",
        ],
        "errors": [],
    }
    environment: dict[str, Any] = {
        "background_mask_ref": background_mask_ref,
        "depth_ref": None,
        "luminance_ref": luminance_ref,
        "exposure_change_ref": exposure_ref,
        "white_balance_proxy_ref": white_balance_ref,
        "blur_ref": blur_ref,
        "occluder_tracks": [],
    }
    emitted = {mask_uri: mask_path, metrics_uri: metrics_path, report_uri: report_path}
    return environment, emitted, report_ref, extension, quality


__all__ = [
    "EnvironmentPhotometryConfig",
    "EnvironmentPhotometryError",
    "run_environment_photometry",
]
