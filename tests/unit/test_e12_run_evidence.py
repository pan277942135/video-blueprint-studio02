from __future__ import annotations

import hashlib

from packages.pipeline_core.e12_run_evidence import collect_surviving_outputs


def test_collect_surviving_outputs_hashes_only_existing_files(tmp_path) -> None:
    blueprint = tmp_path / "blueprint.json"
    bundle = tmp_path / "bundle.zip"
    missing = tmp_path / "acceptance.json"
    blueprint.write_bytes(b'{"ok": true}\n')
    bundle.write_bytes(b"bundle-bytes")

    evidence = collect_surviving_outputs(
        {
            "blueprint": blueprint,
            "bundle": bundle,
            "acceptance_report": missing,
        }
    )

    assert set(evidence) == {"blueprint", "bundle"}
    assert evidence["blueprint"] == {
        "path": str(blueprint),
        "size_bytes": blueprint.stat().st_size,
        "sha256": hashlib.sha256(blueprint.read_bytes()).hexdigest(),
    }
    assert evidence["bundle"]["sha256"] == hashlib.sha256(bundle.read_bytes()).hexdigest()
