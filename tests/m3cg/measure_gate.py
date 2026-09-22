"""Development-only timing/RSS wrapper for the two required shared-layer gates."""
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
GATES = ("test-m3cg-native.sh", "test-m3cg-package.sh")


def main():
    destination = Path(sys.argv[1])
    destination.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    results = []
    for gate in GATES:
        before = time.monotonic()
        result = subprocess.run(["/usr/bin/bash", str(ROOT / "scripts" / gate)], cwd=ROOT)
        results.append({"gate": gate, "seconds": time.monotonic() - before,
                        "exit_code": result.returncode})
        report = {"gates": results, "seconds": time.monotonic() - started,
                  "max_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                  "rss_scope": "maximum child-process RSS on Linux, not summed memory"}
        (destination / "gate-measurements.json").write_text(json.dumps(report, indent=2) + "\n")
        if result.returncode:
            return result.returncode
    print("M3-CG gate measurements: " + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
