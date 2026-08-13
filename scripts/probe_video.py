import json
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from packages.pipeline_core.media_probe import compute_sha256, probe_media


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: probe_video.py <video_path>"}))
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(json.dumps({"error": f"File not found: {video_path}"}))
        sys.exit(1)

    try:
        sha256_hash = compute_sha256(video_path)
        probe_res = probe_media(video_path)
        data = {
            "file_name": os.path.basename(probe_res.video_path),
            "sha256": sha256_hash,
            "mime_type": "video/mp4",
            "file_size_bytes": probe_res.file_size_bytes,
            "duration_us": probe_res.duration_us,
            "width": probe_res.width,
            "height": probe_res.height,
            "fps_avg": probe_res.fps_avg,
            "start_pts_us": probe_res.start_pts_us,
        }
        print(json.dumps(data))
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
