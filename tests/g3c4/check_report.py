"""Compare a K1 report with the exact accepted G3-C4-N inventory."""
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
baseline = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
actual = Path(sys.argv[2]).read_bytes()
rows = json.loads((root / "tests/g3c4/expected/provider_metadata.json").read_text(encoding="utf-8"))
for name, operations, shapes in rows:
    baseline["entries"].append({
        "constraints": {
            "devices": ["cpu"],
            "dtypes": ["f32"],
            "operations": ["g3c4." + operation for operation in operations],
            "shape_ranges": [[[extent, extent] for extent in shape] for shape in sorted(shapes)],
        },
        "deterministic": True,
        "evidence": "G3-C4:cpu-f32-c4-forward-v1",
        "implementation": "g3c4.cpu-f32.serial",
        "name": "g3c4." + name,
        "status": "verified",
        "version": "1.0",
    })
baseline["entries"].sort(key=lambda row: row["name"])
baseline["provider_abi"] = {"major": 1, "minor": 0}
expected = (json.dumps(baseline, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
assert expected == actual, "G3-C4-N report differs from the accepted exact inventory"
pair_count = sum(len(operations) * len(shapes) for _, operations, shapes in rows)
assert pair_count == 28
print(f"G3-C4-N provider report: {len(actual)} bytes, seven capabilities, {pair_count} operation/row pairs")
