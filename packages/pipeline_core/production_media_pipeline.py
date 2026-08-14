from __future__ import annotations

from typing import Any

from packages.pipeline_core.e9_media_pipeline import run_e9_media_pipeline


def run_production_media_pipeline(
    job_id: str,
    video_file_name: str,
    video_sha256: str,
    video_path: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the highest implemented Blueprint pipeline.

    E4-E9 stages remain opt-in and fail-closed through their existing
    environment gates. When no higher-stage configuration is present this
    preserves the certified E1-E3.2 behavior of the underlying pipeline.
    """
    return run_e9_media_pipeline(
        job_id=job_id,
        video_file_name=video_file_name,
        video_sha256=video_sha256,
        video_path=video_path,
    )


__all__ = ["run_production_media_pipeline"]
