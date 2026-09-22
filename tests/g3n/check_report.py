"""Independent accepted-contract report inventory; development-only."""
import json
from pathlib import Path
import sys

baseline = json.loads(Path(sys.argv[1]).read_text())
actual_bytes = Path(sys.argv[2]).read_bytes()
rows = [
    ("causal-attention-forward", ["causal-attention.forward"], [[1, 2, 2, 1, 1, 2]]),
    ("embedding-forward", ["embedding.forward"], [[1, 1, 256, 4], [1, 1, 2, 4]]),
    ("head-layout-forward", ["heads.merge.forward", "heads.split.forward"], [[1, 1, 2, 2]]),
    ("layer-norm-forward", ["layer-norm.forward"], [[1, 1, 4]]),
    ("linear-forward", ["linear.forward-no-bias"], [[1, 1, 4, 4], [1, 1, 4, 8], [1, 1, 8, 4], [1, 1, 4, 256]]),
    ("residual-forward", ["residual.forward"], [[1, 1, 4]]),
]
for name, operations, shapes in rows:
    baseline["entries"].append({
        "constraints": {"devices": ["cpu"], "dtypes": ["f32"],
                        "operations": ["g3n." + op for op in operations],
                        "shape_ranges": [[[d, d] for d in row] for row in sorted(shapes)]},
        "deterministic": True, "evidence": "G3-N:cpu-f32-c2-forward-v1",
        "implementation": "g3n.cpu-f32.serial", "name": "g3n." + name,
        "status": "verified", "version": "1.0",
    })
baseline["entries"].sort(key=lambda row: row["name"])
baseline["provider_abi"] = {"major": 1, "minor": 0}
expected = (json.dumps(baseline, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
assert expected == actual_bytes, "G3-N report differs from exact accepted six-capability/eleven-pair inventory"
assert sum(len(ops) * len(shapes) for _, ops, shapes in rows) == 11
print(f"G3-N provider report: {len(actual_bytes)} bytes, six exact capabilities, eleven operation/row pairs")
