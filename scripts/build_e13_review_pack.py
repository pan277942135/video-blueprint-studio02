from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.pipeline_core.e13_review_pack import build_review_pack


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build side-by-side source-versus-evidence frames for manual E13 perceptual review."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    return parser.parse_args()


def main() -> int:
    args = _args()
    manifest = build_review_pack(
        video_path=pathlib.Path(args.video).resolve(),
        bundle_path=pathlib.Path(args.bundle).resolve(),
        output_dir=pathlib.Path(args.output_dir).resolve(),
        max_frames=args.max_frames,
        confidence_threshold=args.confidence_threshold,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
