import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any


class MediaProbeError(Exception):
    """Base exception for media probe operations."""


class FFprobeNotFoundError(MediaProbeError):
    """Raised when ffprobe executable is missing from system PATH."""


class InvalidMediaError(MediaProbeError):
    """Raised when file is missing, unparseable, or not a valid media file."""


class NoVideoStreamError(MediaProbeError):
    """Raised when no video stream is present in the media file."""


class TimingIndeterminateError(MediaProbeError):
    """Raised when video timing or frame rate cannot be reliably determined."""


def compute_sha256(file_path: str) -> str:
    """Compute stable SHA-256 hash for a local file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def _parse_fraction(frac_str: str | None) -> tuple[int | None, int | None, float | None]:
    if not frac_str or "/" not in frac_str:
        return None, None, None
    try:
        num_s, den_s = frac_str.split("/")
        num, den = int(num_s), int(den_s)
        if den == 0:
            return None, None, None
        return num, den, num / den
    except (ValueError, ZeroDivisionError):
        return None, None, None


def _parse_rotation(vstream: dict[str, Any]) -> int:
    rotation = 0
    side_data_list = vstream.get("side_data_list", [])
    for sd in side_data_list:
        if "rotation" in sd:
            try:
                rotation = int(sd["rotation"])
                break
            except ValueError:
                pass

    if rotation == 0 and "tags" in vstream:
        tags = vstream["tags"]
        rot_val = tags.get("rotate") or tags.get("rotation")
        if rot_val is not None:
            try:
                rotation = int(rot_val)
            except ValueError:
                pass

    normalized = (rotation % 360 + 360) % 360
    if normalized in (0, 90, 180, 270):
        return normalized
    return min([0, 90, 180, 270], key=lambda x: abs(x - normalized))


def _parse_bit_depth(vstream: dict[str, Any]) -> int | None:
    for key in ("bits_per_raw_sample", "bits_per_component"):
        val = vstream.get(key)
        if val is not None and str(val).isdigit():
            bd = int(val)
            if bd > 0:
                return bd

    pix_fmt = str(vstream.get("pix_fmt", ""))
    if "10" in pix_fmt:
        return 10
    if "12" in pix_fmt:
        return 12
    if "16" in pix_fmt:
        return 16
    if "yuv" in pix_fmt or "rgb" in pix_fmt:
        return 8
    return None


@dataclass
class MediaProbeResult:
    video_path: str
    container: str
    file_size_bytes: int
    duration_us: int
    width: int
    height: int
    rotation_deg: int
    video_codec: str
    pixel_format: str | None
    bit_depth: int | None
    color_primaries: str | None
    color_transfer: str | None
    color_space: str | None
    avg_frame_rate: str
    r_frame_rate: str
    fps_avg: float
    fps_nominal: float | None
    source_frame_count: int | None
    variable_frame_rate: bool
    display_aspect_ratio: str | None
    pixel_aspect_ratio: str | None
    video_stream_index: int
    start_pts: int | None
    start_pts_us: int
    source_time_base: str | None
    has_audio: bool
    audio_codec: str | None
    audio_sample_rate_hz: int | None
    audio_channels: int | None
    frame_timestamps: list[float] = field(default_factory=list)

    def to_source_video_dict(self, file_name: str, sha256_hash: str) -> dict[str, Any]:
        ext = os.path.splitext(file_name)[1].lower()
        if ext == ".mp4" or "mp4" in self.container:
            mime_type = "video/mp4"
        elif ext == ".mov" or "mov" in self.container:
            mime_type = "video/quicktime"
        elif ext == ".mkv" or "matroska" in self.container:
            mime_type = "video/x-matroska"
        elif ext == ".webm" or "webm" in self.container:
            mime_type = "video/webm"
        else:
            mime_type = f"video/{self.container.split(',')[0]}"

        return {
            "file_name": file_name,
            "mime_type": mime_type,
            "container": self.container,
            "file_size_bytes": self.file_size_bytes,
            "sha256": sha256_hash,
            "duration_us": self.duration_us,
            "width": self.width,
            "height": self.height,
            "display_aspect_ratio": self.display_aspect_ratio,
            "pixel_aspect_ratio": self.pixel_aspect_ratio,
            "rotation_deg": self.rotation_deg,
            "video_codec": self.video_codec,
            "pixel_format": self.pixel_format,
            "bit_depth": self.bit_depth,
            "color_primaries": self.color_primaries,
            "color_transfer": self.color_transfer,
            "color_space": self.color_space,
            "fps_avg": round(self.fps_avg, 3),
            "fps_nominal": round(self.fps_nominal, 3) if self.fps_nominal is not None else None,
            "variable_frame_rate": self.variable_frame_rate,
            "source_frame_count": self.source_frame_count,
            "has_audio": self.has_audio,
            "audio_codec": self.audio_codec,
            "audio_sample_rate_hz": self.audio_sample_rate_hz,
            "audio_channels": self.audio_channels,
            "metadata_stripped": False
        }

    def to_timebase_dict(self) -> dict[str, Any]:
        num, den, _ = _parse_fraction(self.avg_frame_rate)
        if not num or not den:
            frac = Fraction(self.fps_avg).limit_denominator(100000)
            num, den = frac.numerator, frac.denominator

        if self.source_frame_count is None or self.source_frame_count < 1:
            raise TimingIndeterminateError(
                "Source frame count could not be reliably determined for canonical timebase."
            )

        frame_count = self.source_frame_count
        # Note: This is the canonical nominal frame duration for the timebase,
        # calculated from the rational frame rate, not the variable frame-by-frame duration.
        frame_duration_us = round(1_000_000.0 * den / num, 3)
        end_pts_us = self.start_pts_us + self.duration_us

        return {
            "normalized_to_cfr": False,
            "fps_num": num,
            "fps_den": den,
            "frame_count": frame_count,
            "frame_duration_us": frame_duration_us,
            "start_pts_us": self.start_pts_us,
            "end_pts_us": end_pts_us,
            "source_pts_map_ref": None
        }


def probe_media(video_path: str) -> MediaProbeResult:
    """
    Executes ffprobe to extract detailed technical video metadata.
    Raises MediaProbeError subclasses on missing binary, invalid file, missing streams,
    or indeterminate timing/frame rate.
    """
    if not os.path.exists(video_path) or not os.path.isfile(video_path):
        raise InvalidMediaError(f"Video file does not exist: {video_path}")

    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        raise FFprobeNotFoundError("ffprobe executable was not found in PATH.")

    cmd_stream = [
        ffprobe_bin,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path
    ]

    try:
        res = subprocess.run(cmd_stream, capture_output=True, check=True, text=True)
    except subprocess.CalledProcessError as e:
        raise InvalidMediaError(f"ffprobe execution failed on {video_path}: {e.stderr.strip()}") from e

    try:
        data = json.loads(res.stdout)
    except json.JSONDecodeError as e:
        raise InvalidMediaError(f"ffprobe produced invalid JSON output for {video_path}") from e

    fmt = data.get("format", {})
    streams = data.get("streams", [])

    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    if not video_streams:
        raise NoVideoStreamError(f"No video stream found in media file: {video_path}")

    vstream = video_streams[0]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    video_stream_index = int(vstream.get("index", 0))
    file_size_bytes = os.path.getsize(video_path)

    # Duration
    duration_sec: float | None = None
    if "duration" in fmt and fmt["duration"] != "N/A":
        try:
            duration_sec = float(fmt["duration"])
        except ValueError:
            pass
    if duration_sec is None and "duration" in vstream and vstream["duration"] != "N/A":
        try:
            duration_sec = float(vstream["duration"])
        except ValueError:
            pass

    if duration_sec is None or duration_sec <= 0:
        raise InvalidMediaError(f"Unable to determine valid video duration for {video_path}")

    duration_us = round(duration_sec * 1_000_000)

    # Dimensions
    try:
        width = int(vstream.get("width", 0))
        height = int(vstream.get("height", 0))
    except ValueError as e:
        raise InvalidMediaError(f"Invalid dimensions in video stream for {video_path}") from e

    if width <= 0 or height <= 0:
        raise InvalidMediaError(f"Video width/height must be positive: {width}x{height}")

    video_codec = vstream.get("codec_name", "unknown")
    container = fmt.get("format_name", "unknown")

    # Frame rate
    avg_fr_str = vstream.get("avg_frame_rate", "0/0")
    r_fr_str = vstream.get("r_frame_rate", "0/0")

    _, _, fps_avg_calc = _parse_fraction(avg_fr_str)
    _, _, fps_r_calc = _parse_fraction(r_fr_str)

    fps_avg = fps_avg_calc or fps_r_calc
    if not fps_avg or fps_avg <= 0:
        raise TimingIndeterminateError(f"Invalid or missing frame rate for {video_path}: avg={avg_fr_str}, r={r_fr_str}")

    fps_nominal = fps_r_calc or fps_avg

    # Time Base & Start PTS Calculation
    source_time_base = vstream.get("time_base")
    start_pts_raw = vstream.get("start_pts")
    start_pts: int | None = None
    if start_pts_raw is not None and str(start_pts_raw).lstrip("-").isdigit():
        start_pts = int(start_pts_raw)

    start_pts_us = 0
    if start_pts is not None and source_time_base:
        tb_num, tb_den, _ = _parse_fraction(source_time_base)
        if tb_num and tb_den:
            start_pts_us = round(start_pts * (tb_num / tb_den) * 1_000_000)

    if start_pts_us == 0:
        start_time_str = vstream.get("start_time") or fmt.get("start_time")
        if start_time_str and start_time_str != "N/A":
            try:
                start_pts_us = round(float(start_time_str) * 1_000_000)
            except ValueError:
                pass

    # Frame Timestamp Inspection for Conservative VFR & Frame Count
    cmd_frames = [
        ffprobe_bin,
        "-v", "quiet",
        "-print_format", "json",
        "-show_entries", "frame=pkt_pts,pkt_pts_time,best_effort_timestamp_time,pkt_duration_time",
        "-select_streams", "v:0",
        video_path
    ]

    try:
        res_frames = subprocess.run(cmd_frames, capture_output=True, check=True, text=True)
        frames_data: list[dict[str, Any]] = json.loads(res_frames.stdout).get("frames", [])
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        raise TimingIndeterminateError(f"Frame-level ffprobe inspection failed for {video_path}") from e

    if not frames_data:
        raise TimingIndeterminateError(f"No frame data returned from ffprobe inspection for {video_path}")

    # Source frame count priority: nb_frames -> len(frames_data)
    source_frame_count: int | None = None
    if "nb_frames" in vstream and str(vstream["nb_frames"]).isdigit() and int(vstream["nb_frames"]) > 0:
        source_frame_count = int(vstream["nb_frames"])
    else:
        source_frame_count = len(frames_data)

    # Conservative VFR detection
    frame_timestamps: list[float] = []
    for f in frames_data:
        ts_val = f.get("best_effort_timestamp_time") or f.get("pkt_pts_time")
        if ts_val is not None and ts_val != "N/A":
            try:
                frame_timestamps.append(float(ts_val))
            except ValueError:
                pass

    if len(frame_timestamps) < 2:
        raise TimingIndeterminateError(
            f"Insufficient frame timestamps ({len(frame_timestamps)}) for reliable timing inspection in {video_path}"
        )

    deltas = [frame_timestamps[i + 1] - frame_timestamps[i] for i in range(len(frame_timestamps) - 1)]
    if any(d <= 0 for d in deltas):
        raise TimingIndeterminateError(f"Non-positive or non-monotonic frame delta encountered in {video_path}")

    max_delta = max(deltas)
    min_delta = min(deltas)
    # If delta variation > 2ms or fraction divergence > 0.05
    if (max_delta - min_delta) > 0.002 or (fps_avg_calc and fps_r_calc and abs(fps_avg_calc - fps_r_calc) > 0.05):
        vfr = True
    else:
        vfr = False

    # Optional metadata fields
    pixel_format = vstream.get("pix_fmt")
    if pixel_format in ("none", "N/A"):
        pixel_format = None

    bit_depth = _parse_bit_depth(vstream)

    def _clean_str(val: str | None) -> str | None:
        if val in (None, "unknown", "N/A", "unspecified", "none"):
            return None
        return str(val)

    color_primaries = _clean_str(vstream.get("color_primaries"))
    color_transfer = _clean_str(vstream.get("color_transfer"))
    color_space = _clean_str(vstream.get("color_space"))

    dar = _clean_str(vstream.get("display_aspect_ratio"))
    par = _clean_str(vstream.get("sample_aspect_ratio"))

    rotation_deg = _parse_rotation(vstream)

    has_audio = len(audio_streams) > 0
    audio_codec: str | None = None
    audio_sample_rate_hz: int | None = None
    audio_channels: int | None = None

    if has_audio:
        astream = audio_streams[0]
        audio_codec = astream.get("codec_name")
        if "sample_rate" in astream and str(astream["sample_rate"]).isdigit():
            audio_sample_rate_hz = int(astream["sample_rate"])
        if "channels" in astream and str(astream["channels"]).isdigit():
            audio_channels = int(astream["channels"])

    return MediaProbeResult(
        video_path=video_path,
        container=container,
        file_size_bytes=file_size_bytes,
        duration_us=duration_us,
        width=width,
        height=height,
        rotation_deg=rotation_deg,
        video_codec=video_codec,
        pixel_format=pixel_format,
        bit_depth=bit_depth,
        color_primaries=color_primaries,
        color_transfer=color_transfer,
        color_space=color_space,
        avg_frame_rate=avg_fr_str,
        r_frame_rate=r_fr_str,
        fps_avg=fps_avg,
        fps_nominal=fps_nominal,
        source_frame_count=source_frame_count,
        variable_frame_rate=vfr,
        display_aspect_ratio=dar,
        pixel_aspect_ratio=par,
        video_stream_index=video_stream_index,
        start_pts=start_pts,
        start_pts_us=start_pts_us,
        source_time_base=source_time_base,
        has_audio=has_audio,
        audio_codec=audio_codec,
        audio_sample_rate_hz=audio_sample_rate_hz,
        audio_channels=audio_channels,
        frame_timestamps=frame_timestamps
    )
