"""Source-only topology for the exact F image-loader collector leaf."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "native/g3c4_public_g2_f_artifact_source_closure.txt"
CLOSURE = ROOT / "native/g3c4_public_g2_loader_collector_source_closure.txt"
ADDED = (
    "tests/g3c4/f_image_loader_collect.py",
    "tests/g3c4/test_f_image_loader_collect.py",
    "tests/g3c4/check_f_image_loader_collect.py",
    "docs/g3/G3_C4_F_IMAGE_LOADER_COLLECTOR_SOURCE_LEAF.md",
    "native/g3c4_public_g2_loader_collector_source_closure.txt",
)
UNCHANGED = {
    "tests/g3c4/f_artifact_attest.py":
        "5aac55665ad5a1767c54591ec3530667cced310f07948257521f34ed43999e78",
    "tests/g3g_manual/public_g2_runtime.esk":
        "c9f92ca1af857694ecd56e4fe177b52e5183679e61f82accc99e841591aaca91",
    "scripts/common.sh":
        "2749689e6301cad6e489de4b23e0f9b1ec7c9627f47dd1c7007d9ed8e5955e83",
    "toolchain/eshkol.lock":
        "8c168f5d560781245f6d4c73f695e72cbafa413b606d50bc7c36d2119acd9efe",
}


def validate(source: str, base: list[str], closure: list[str],
             originals: dict[str, bytes]) -> None:
    assert len(base) == 222 and closure == base + list(ADDED)
    assert len(closure) == len(set(closure))
    assert all((ROOT / name).is_file() for name in closure)
    for name, expected in UNCHANGED.items():
        assert hashlib.sha256(originals[name]).hexdigest() == expected, name
    ast.parse(source, feature_version=(3, 10))
    for anchor in ("TOOL_PINS_SHA =", "RUNTIME33_SHA =", "TIMEOUT_SECONDS = 120",
                   "STDOUT_LIMIT = 2 * 1024 * 1024", "STDERR_LIMIT = 64 * 1024",
                   "attest.copied_artifacts(", "attest.elf_members(",
                   "attest.resolved_image_libs(", "image_state(",
                   "loader_rows(", "os.WNOWAIT", "os.wait4(",
                   "start_new_session=True", "signal_owned(",
                   "unadmitted loader dependency", "collector post-publication"):
        assert anchor in source, anchor
    assert "subprocess.run(" not in source
    assert "--tool-pins" in source and "--runtime-pins" in source
    assert "args.tool_pins == TOOL_PINS" in source
    assert "args.runtime_pins == RUNTIME_PINS" in source


def main() -> None:
    validate((ROOT / ADDED[0]).read_text(), BASE.read_text().splitlines(),
             CLOSURE.read_text().splitlines(),
             {name: (ROOT / name).read_bytes() for name in UNCHANGED})
    print("F image-loader collector source contract: PASS")


if __name__ == "__main__":
    main()
