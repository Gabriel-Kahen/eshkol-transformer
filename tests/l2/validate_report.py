from __future__ import annotations

import json
import sys
from pathlib import Path


def main(path: Path) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    entries = {entry["name"]: entry for entry in value["entries"]}
    entry = entries["kernel.indexed-cross-entropy"]
    assert value["format"] == "eshkol-kernel-capabilities"
    assert value["provider_abi"] == {"major": 1, "minor": 0}
    assert entry["status"] == "verified"
    assert entry["deterministic"] is True
    assert entry["implementation"] == "eshkol-transformer-l2-cpu"
    assert entry["version"] == "1.0.0"
    assert entry["evidence"] == "L2:cpu-f32-indexed-cross-entropy-v1"
    assert entry["constraints"] == {
        "devices": ["cpu"],
        "dtypes": ["f32"],
        "operations": [
            "indexed-cross-entropy.backward",
            "indexed-cross-entropy.forward",
        ],
        "shape_ranges": [[[1, 1664510], [1, 1664510], [1, 1664510]]],
    }
    for name in ("tensor.f32", "tensor.i64", "tensor.contiguous", "autodiff.reverse"):
        assert entries[name]["status"] == "unverified"
        assert entries[name]["constraints"] == {
            "devices": [], "dtypes": [], "operations": [], "shape_ranges": []
        }
    print("L2 capability report: PASS")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
