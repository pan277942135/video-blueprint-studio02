import os
from typing import Any

from packages.pipeline_core.media_probe import InvalidMediaError
from packages.pipeline_core.mock_pipeline import run_deterministic_mock_pipeline


def run_real_media_pipeline(
    job_id: str,
    video_file_name: str,
    video_sha256: str,
    video_path: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the E1 real-media path with no mock-media fallback.

    E1 still reuses the deterministic non-CV blueprint scaffold while real CV
    modules are not implemented yet, but source metadata, normalization,
    timeline, hashes, and exported artifacts MUST come from the uploaded file.
    Missing or invalid media is a hard failure; it must never degrade to E0
    sample/default metadata.
    """
    if not video_path:
        raise InvalidMediaError("Real-media analysis requires a persisted uploaded video path")
    if not os.path.isfile(video_path):
        raise InvalidMediaError(f"Uploaded video path does not exist: {video_path}")

    return run_deterministic_mock_pipeline(
        job_id=job_id,
        video_file_name=video_file_name,
        video_sha256=video_sha256,
        video_path=video_path,
    )
