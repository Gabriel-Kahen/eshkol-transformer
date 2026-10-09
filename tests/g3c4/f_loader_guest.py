"""Test-only copied-F preparation and image-loader observation; no public gate.

The outer ROOT capsule must authenticate the image, mounts and one-shot launch.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from tests.g3c4 import f_artifact_attest as attest
from tests.g3c4 import f_image_loader_collect as collector


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-admission-sha256", required=True)
    args = p.parse_args()
    argv = ["prepare", "--f-output", "/f-output", "--compiler-source",
            "/workspace", "--copy-build", "/out/build-off",
            "--candidate-lock", "/bundle/public-g2-6803.lock",
            "--offline-review", "/bundle/offline/SHA256SUMS-ROOT-REVIEW",
            "--project-source", "/stage/candidate", "--outer-stage-child",
            "--project-checkout", "/source",
            "--project-admission", "/bundle/public-g2-source-admission.json",
            "--project-admission-sha256", args.source_admission_sha256,
            "--evidence", "/out/evidence"]
    original = sys.argv
    try:
        sys.argv = ["f_artifact_attest.py", *argv]
        attest.main()
    finally:
        sys.argv = original
    collector.collect(Path("/out/build-off/eshkol-run"),
                      collector.TOOL_PINS, collector.RUNTIME_PINS,
                      Path("/out/evidence/loader"))


if __name__ == "__main__":
    main()
