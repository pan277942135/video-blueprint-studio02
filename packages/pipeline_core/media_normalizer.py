import json
import os
import shutil
import statistics
import subprocess
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from packages.pipeline_core.media_probe import (
    MediaProbeResult,
    compute_sha256,
    probe_media,
)


class MediaNormalizationError(Exception):
    """Base exception for media normalization operations."""


class FFmpegNotFoundError(MediaNormalizationError):
    """Raised when ffmpeg executable is missing from system PATH."""


class NormalizationFailedError(MediaNormalizationError):
    """Raised when ffmpeg normalization fails or produces empty output."""


class NormalizationValidationError(MediaNormalizationError):
    """Raised when normalized output fails post-processing validation."""


@dataclass
class MediaNormalizationResult:
    normalized_video_path: str
    normalized_video_sha256: str
    fps_num: int
    fps_den: int
    fps: float
    frame_count: int
    frame_duration_us: float
    start_pts_us: int  # Always 0
    end_pts_us: int
    source_pts_map_path: str
    source_pts_map_sha256: str
    source_variable_frame_rate: bool
    audio_preserved: bool
    audio_duration_us: int | None
    video_duration_us: int
    audio_video_drift_us: int | None
    codec: str
    pixel_format: str
    ffmpeg_command: list[str]
    normalization_profile: str
    mapping_error_max_us: float
    mapping_error_mean_us: float


STANDARD_RATIONAL_FPS = [
    (24000, 1001),
    (24, 1),
    (25, 1),
    (30000, 1001),
    (30, 1),
    (50, 1),
    (60000, 1001),
    (60, 1),
]


def determine_target_fps(
    probe_result: MediaProbeResult,
    analysis_fps_max: int = 30
) -> tuple[int, int]:
    """
    Determines deterministic target rational FPS (fps_num, fps_den) for CFR normalization.
    """
    if not probe_result.variable_frame_rate:
        # CFR Source: keep exact rational if possible
        num, den = None, None
        for fr_str in (probe_result.r_frame_rate, probe_result.avg_frame_rate):
            if fr_str and "/" in fr_str:
                try:
                    n_s, d_s = fr_str.split("/")
                    n_val, d_val = int(n_s), int(d_s)
                    if d_val > 0 and n_val > 0:
                        num, den = n_val, d_val
                        break
                except ValueError:
                    pass

        if num is None or den is None:
            frac = Fraction(probe_result.fps_avg).limit_denominator(1001)
            num, den = frac.numerator, frac.denominator

        source_fps_float = num / den
        if source_fps_float > analysis_fps_max:
            # Scale down if integer factor e.g. 60/1 -> 30/1, 60000/1001 -> 30000/1001
            if num % 2 == 0 and (num // 2) / den <= analysis_fps_max:
                return num // 2, den
            return analysis_fps_max, 1
        return num, den

    # VFR Source: derive nominal FPS using median frame delta
    if probe_result.frame_timestamps and len(probe_result.frame_timestamps) >= 2:
        deltas = [
            probe_result.frame_timestamps[i + 1] - probe_result.frame_timestamps[i]
            for i in range(len(probe_result.frame_timestamps) - 1)
        ]
        med_delta = statistics.median(deltas)
        if med_delta <= 0:
            nominal_fps = probe_result.fps_avg
        else:
            nominal_fps = 1.0 / med_delta
    else:
        nominal_fps = probe_result.fps_avg

    # Match closest standard candidate
    best_cand = None
    min_diff = float("inf")
    for num_cand, den_cand in STANDARD_RATIONAL_FPS:
        cand_fps = num_cand / den_cand
        diff = abs(cand_fps - nominal_fps)
        if diff < min_diff:
            min_diff = diff
            best_cand = (num_cand, den_cand)

    if best_cand is not None and min_diff < 0.5:
        target_num, target_den = best_cand
    else:
        frac = Fraction(nominal_fps).limit_denominator(1001)
        target_num, target_den = frac.numerator, frac.denominator

    target_fps_float = target_num / target_den
    if target_fps_float > analysis_fps_max:
        if target_num % 2 == 0 and (target_num // 2) / target_den <= analysis_fps_max:
            return target_num // 2, target_den
        return analysis_fps_max, 1

    return target_num, target_den


def generate_source_pts_map(
    norm_frame_count: int,
    fps_num: int,
    fps_den: int,
    orig_start_pts_us: int,
    source_frame_timestamps: list[float]
) -> tuple[np.ndarray, float, float]:
    """
    Generates [norm_frame_count, 2] int64 matrix of [normalized_pts_us, source_pts_us].
    Returns (matrix, max_error_us, mean_error_us).
    """
    source_ts_us = [round(t * 1_000_000) for t in source_frame_timestamps]
    source_ts_us.sort()

    rows: list[list[int]] = []
    errors: list[float] = []

    last_chosen_source_pts = source_ts_us[0] if source_ts_us else orig_start_pts_us

    for n in range(norm_frame_count):
        norm_pts_us = round(n * 1_000_000 * fps_den / fps_num)
        desired_source_pts_us = orig_start_pts_us + norm_pts_us

        if source_ts_us:
            best_ts = min(source_ts_us, key=lambda ts: abs(ts - desired_source_pts_us))
        else:
            best_ts = desired_source_pts_us

        if best_ts < last_chosen_source_pts:
            chosen_source_pts = last_chosen_source_pts
        else:
            chosen_source_pts = best_ts

        last_chosen_source_pts = chosen_source_pts

        err = abs(chosen_source_pts - desired_source_pts_us)
        errors.append(float(err))
        rows.append([norm_pts_us, chosen_source_pts])

    matrix = np.array(rows, dtype=np.int64)
    max_err = max(errors) if errors else 0.0
    mean_err = (sum(errors) / len(errors)) if errors else 0.0

    return matrix, max_err, mean_err


def probe_stream_durations_us(file_path: str) -> tuple[int, int | None]:
    """
    Returns (video_duration_us, audio_duration_us) for media file.
    Probes video stream (v:0) and audio stream (a:0) independently.
    """
    ffprobe_bin = shutil.which("ffprobe") or "ffprobe"
    cmd = [
        ffprobe_bin,
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        file_path
    ]
    res = subprocess.run(cmd, capture_output=True, check=True, text=True)
    data = json.loads(res.stdout)
    streams = data.get("streams", [])

    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    video_dur_us: int | None = None
    if video_streams:
        v_s = video_streams[0]
        if "duration" in v_s and v_s["duration"] != "N/A":
            try:
                video_dur_us = round(float(v_s["duration"]) * 1_000_000)
            except ValueError:
                pass

    audio_dur_us: int | None = None
    if audio_streams:
        a_s = audio_streams[0]
        if "duration" in a_s and a_s["duration"] != "N/A":
            try:
                audio_dur_us = round(float(a_s["duration"]) * 1_000_000)
            except ValueError:
                pass

    if video_dur_us is None and video_streams:
        cmd_v = [
            ffprobe_bin, "-v", "quiet", "-print_format", "json",
            "-show_entries", "frame=pkt_pts_time,pkt_duration_time",
            "-select_streams", "v:0", file_path
        ]
        res_v = subprocess.run(cmd_v, capture_output=True, check=False, text=True)
        if res_v.returncode == 0:
            v_frames = json.loads(res_v.stdout).get("frames", [])
            if v_frames:
                last_f = v_frames[-1]
                pts = float(last_f.get("pkt_pts_time", 0))
                dur = float(last_f.get("pkt_duration_time", 0))
                video_dur_us = round((pts + dur) * 1_000_000)

    if audio_streams and audio_dur_us is None:
        cmd_a = [
            ffprobe_bin, "-v", "quiet", "-print_format", "json",
            "-show_entries", "packet=pts_time,duration_time",
            "-select_streams", "a:0", file_path
        ]
        res_a = subprocess.run(cmd_a, capture_output=True, check=False, text=True)
        if res_a.returncode == 0:
            a_pkts = json.loads(res_a.stdout).get("packets", [])
            if a_pkts:
                last_p = a_pkts[-1]
                pts = float(last_p.get("pts_time", 0))
                dur = float(last_p.get("duration_time", 0))
                audio_dur_us = round((pts + dur) * 1_000_000)

    if video_dur_us is None:
        norm_p = probe_media(file_path)
        video_dur_us = norm_p.duration_us

    return video_dur_us, audio_dur_us


def validate_normalization(
    normalized_video_path: str,
    probe_result: MediaProbeResult,
    target_fps_num: int,
    target_fps_den: int,
    pts_map_matrix: np.ndarray,
    max_av_drift_us: int = 50000
) -> tuple[MediaProbeResult, int, int | None, int | None]:
    """
    Re-probes normalized video and validates all contract/normalization invariants.
    Returns (norm_probe, video_duration_us, audio_duration_us, audio_video_drift_us).
    """
    if not os.path.exists(normalized_video_path):
        raise NormalizationValidationError(f"Normalized output file missing: {normalized_video_path}")

    if os.path.getsize(normalized_video_path) == 0:
        raise NormalizationValidationError(f"Normalized output file is 0 bytes: {normalized_video_path}")

    norm_probe = probe_media(normalized_video_path)

    if norm_probe.width != probe_result.width:
        raise NormalizationValidationError(
            f"Width mismatch: normalized {norm_probe.width} != source {probe_result.width}"
        )

    if norm_probe.height != probe_result.height:
        raise NormalizationValidationError(
            f"Height mismatch: normalized {norm_probe.height} != source {probe_result.height}"
        )

    if norm_probe.variable_frame_rate:
        raise NormalizationValidationError("Normalized output detected as VFR, expected CFR")

    if norm_probe.start_pts_us != 0:
        raise NormalizationValidationError(
            f"Normalized video start_pts_us is {norm_probe.start_pts_us}, expected 0"
        )

    if norm_probe.source_frame_count is None or norm_probe.source_frame_count < 1:
        raise NormalizationValidationError("Normalized video source_frame_count must be >= 1")

    if norm_probe.source_frame_count != pts_map_matrix.shape[0]:
        raise NormalizationValidationError(
            f"Frame count mismatch: normalized video has {norm_probe.source_frame_count} frames "
            f"but source_pts_map has {pts_map_matrix.shape[0]} rows"
        )

    target_frac = Fraction(target_fps_num, target_fps_den)
    actual_frac = None
    for fr_str in (norm_probe.r_frame_rate, norm_probe.avg_frame_rate):
        if fr_str and "/" in fr_str:
            try:
                n_s, d_s = fr_str.split("/")
                n_v, d_v = int(n_s), int(d_s)
                if d_v > 0 and n_v > 0:
                    actual_frac = Fraction(n_v, d_v)
                    break
            except ValueError:
                pass

    if actual_frac is not None:
        if actual_frac != target_frac:
            raise NormalizationValidationError(
                f"Normalized video rational FPS {actual_frac} does not match target {target_frac}"
            )
    else:
        target_fps_float = float(target_frac)
        if abs(norm_probe.fps_avg - target_fps_float) > 1e-4:
            raise NormalizationValidationError(
                f"Normalized video FPS {norm_probe.fps_avg:.5f} diverges from target {target_fps_float:.5f}"
            )

    video_dur_us, audio_dur_us = probe_stream_durations_us(normalized_video_path)

    if probe_result.has_audio:
        if not norm_probe.has_audio:
            raise NormalizationValidationError("Source has audio but normalized video is missing audio stream")
        if audio_dur_us is None:
            raise NormalizationValidationError("Unable to determine audio stream duration in normalized output")

        drift = abs(audio_dur_us - video_dur_us)
        frame_dur_us = 1_000_000 * target_fps_den / target_fps_num
        max_allowed_drift = max(max_av_drift_us, round(frame_dur_us))
        if drift > max_allowed_drift:
            raise NormalizationValidationError(
                f"Audio/video drift {drift}us exceeds threshold {max_allowed_drift}us"
            )
    else:
        if norm_probe.has_audio:
            raise NormalizationValidationError("Source has no audio but normalized video contains audio stream")
        drift = None

    return norm_probe, video_dur_us, audio_dur_us, drift


def normalize_media_to_cfr(
    video_path: str,
    probe_result: MediaProbeResult,
    output_dir: str,
    analysis_fps_max: int = 30,
    video_codec: str = "libx264",
    preset: str = "veryfast",
    crf: int = 12,
    max_av_drift_us: int = 50000
) -> MediaNormalizationResult:
    """
    Executes FFmpeg CFR normalization on source video and builds deterministic source PTS map.
    """
    if not shutil.which("ffmpeg"):
        raise FFmpegNotFoundError("ffmpeg executable not found in system PATH")

    if not os.path.exists(video_path):
        raise NormalizationFailedError(f"Input video file does not exist: {video_path}")

    norm_dir = os.path.join(output_dir, "artifacts", "normalized")
    ts_dir = os.path.join(output_dir, "artifacts", "timeseries")
    os.makedirs(norm_dir, exist_ok=True)
    os.makedirs(ts_dir, exist_ok=True)

    normalized_video_path = os.path.join(norm_dir, "analysis_cfr.mp4")
    source_pts_map_path = os.path.join(ts_dir, "source_pts_map.npz")

    fps_num, fps_den = determine_target_fps(probe_result, analysis_fps_max)
    fps_float = fps_num / fps_den

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"setpts=PTS-STARTPTS,fps={fps_num}/{fps_den}:round=near",
        "-fps_mode", "cfr",
        "-c:v", video_codec,
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p"
    ]
    if probe_result.has_audio:
        cmd.extend([
            "-af", "asetpts=PTS-STARTPTS,aresample=async=1:first_pts=0",
            "-c:a", "aac",
            "-b:a", "128k"
        ])
    else:
        cmd.append("-an")

    cmd.append(normalized_video_path)

    try:
        subprocess.run(cmd, capture_output=True, check=True, text=True)
    except subprocess.CalledProcessError as e:
        raise NormalizationFailedError(
            f"FFmpeg normalization failed for {video_path}: {e.stderr}"
        ) from e

    if not os.path.exists(normalized_video_path) or os.path.getsize(normalized_video_path) == 0:
        raise NormalizationFailedError(f"FFmpeg produced empty or missing file at {normalized_video_path}")

    norm_probe_pre = probe_media(normalized_video_path)
    if norm_probe_pre.source_frame_count is None or norm_probe_pre.source_frame_count < 1:
        raise NormalizationFailedError(
            f"FFmpeg normalization resulted in invalid frame_count: {norm_probe_pre.source_frame_count}"
        )
    frame_count = norm_probe_pre.source_frame_count

    pts_map_matrix, max_err_us, mean_err_us = generate_source_pts_map(
        norm_frame_count=frame_count,
        fps_num=fps_num,
        fps_den=fps_den,
        orig_start_pts_us=probe_result.start_pts_us,
        source_frame_timestamps=probe_result.frame_timestamps
    )

    np.savez_compressed(source_pts_map_path, source_pts_map=pts_map_matrix)

    norm_probe, video_dur_us, audio_dur_us, drift_us = validate_normalization(
        normalized_video_path=normalized_video_path,
        probe_result=probe_result,
        target_fps_num=fps_num,
        target_fps_den=fps_den,
        pts_map_matrix=pts_map_matrix,
        max_av_drift_us=max_av_drift_us
    )

    norm_video_sha256 = compute_sha256(normalized_video_path)
    pts_map_sha256 = compute_sha256(source_pts_map_path)

    frame_duration_us = 1_000_000 * fps_den / fps_num
    end_pts_us = round((frame_count - 1) * frame_duration_us)

    return MediaNormalizationResult(
        normalized_video_path=normalized_video_path,
        normalized_video_sha256=norm_video_sha256,
        fps_num=fps_num,
        fps_den=fps_den,
        fps=fps_float,
        frame_count=frame_count,
        frame_duration_us=frame_duration_us,
        start_pts_us=0,
        end_pts_us=end_pts_us,
        source_pts_map_path=source_pts_map_path,
        source_pts_map_sha256=pts_map_sha256,
        source_variable_frame_rate=probe_result.variable_frame_rate,
        audio_preserved=probe_result.has_audio,
        audio_duration_us=audio_dur_us,
        video_duration_us=video_dur_us,
        audio_video_drift_us=drift_us,
        codec=norm_probe.video_codec,
        pixel_format=norm_probe.pixel_format or "yuv420p",
        ffmpeg_command=cmd,
        normalization_profile="analysis_cfr_v1",
        mapping_error_max_us=max_err_us,
        mapping_error_mean_us=mean_err_us
    )

