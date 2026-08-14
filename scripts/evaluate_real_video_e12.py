from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import zipfile

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.pipeline_core.artifact_integrity import verify_bundle_zip
from packages.pipeline_core.real_video_acceptance import summarize_blueprint_acceptance


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an E12 evidence-first acceptance report from a Video Blueprint Bundle"
    )
    parser.add_argument("--bundle", required=True, help="Path to bundle.zip")
    parser.add_argument("--output", default="e12_real_video_acceptance.json")
    parser.add_argument(
        "--low-score-threshold",
        type=float,
        default=0.5,
        help="Review heuristic only; scores below this are flagged but do not define perceptual accuracy",
    )
    return parser.parse_args()


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = _args()
    bundle = pathlib.Path(args.bundle).resolve()
    output = pathlib.Path(args.output).resolve()
    if not bundle.is_file():
        raise SystemExit(f"missing bundle: {bundle}")
    if not 0.0 <= args.low_score_threshold <= 1.0:
        raise SystemExit("--low-score-threshold must be in [0,1]")

    integrity = verify_bundle_zip(str(bundle))
    try:
        with zipfile.ZipFile(bundle, "r") as archive:
            blueprint = json.loads(archive.read("blueprint.json"))
    except (zipfile.BadZipFile, KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read blueprint.json from bundle: {exc}") from exc
    if not isinstance(blueprint, dict):
        raise SystemExit("blueprint.json must contain an object")

    report = summarize_blueprint_acceptance(
        blueprint,
        integrity=integrity,
        low_score_threshold=args.low_score_threshold,
    )
    report["artifact_integrity"]["zip_entry_count"] = integrity.get("zip_entry_count")
    report["bundle"] = {
        "path": str(bundle),
        "size_bytes": bundle.stat().st_size,
        "sha256": _sha256_file(bundle),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["machine_gate"] != "failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
