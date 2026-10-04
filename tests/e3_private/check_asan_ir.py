"""Verify exact streamed E3 private ASan IR and generated object coverage."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tests.shared.prepare_eshkol_asan_ir import real_directory, regular, scan_pair


REPORT = re.compile(rb"^__asan_report_(?:load|store)(?:[0-9]+|_n) U[ \t]*$", re.M)
INIT = re.compile(rb"^__asan_init U[ \t]*$", re.M)


def verify(raw: Path, derived: Path, receipt: Path, object_file: Path,
           imports: Path) -> None:
    for path in (raw, derived, receipt, object_file, imports):
        real_directory(path.parent)
        regular(path)
    expected = scan_pair(raw, derived)
    encoded = (json.dumps(expected, sort_keys=True, separators=(",", ":"))
               + "\n").encode("ascii")
    if receipt.read_bytes() != encoded:
        raise ValueError("E3 ASan IR receipt differs from exact streaming rewrite")
    symbols = subprocess.run(
        ["nm", "-u", "--format=posix", str(object_file)],
        capture_output=True, check=False,
    )
    if symbols.returncode != 0 or symbols.stderr or symbols.stdout != imports.read_bytes():
        raise ValueError("E3 generated object imports differ from raw nm evidence")
    if not REPORT.search(symbols.stdout) or not INIT.search(symbols.stdout):
        raise ValueError("E3 generated object lacks address-check report imports")


def main() -> None:
    if len(sys.argv) != 6:
        raise SystemExit("usage: check_asan_ir.py RAW_LL ASAN_LL RECEIPT OBJECT IMPORTS")
    verify(*(Path(argument) for argument in sys.argv[1:]))


if __name__ == "__main__":
    main()
