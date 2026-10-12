#!/usr/bin/env python3
"""Check emitted companion completeness and bytewise direct-build equivalence."""
import hashlib
import filecmp
import json
from pathlib import Path
import sys

MODES = ("normal", "sanitize")
FIXTURES = ("terminal", "feature-off", "parity-original", "parity-overlay",
            "unmarked", "decode-first", "decode-second", "precommit", "postzero")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify(build: Path) -> dict:
    if not build.is_dir() or build.is_symlink():
        raise ValueError("missing build tree")
    hashes = {}
    for mode in MODES:
        directory = build / mode
        for fixture in FIXTURES:
            emitted = directory / "emitted"
            prefix = emitted / f"{fixture}.raw.o"
            paths = [prefix, Path(str(prefix) + ".bc"), Path(str(prefix) + ".ll"),
                     emitted / f"{fixture}.d", emitted / f"{fixture}.map",
                     emitted / f"{fixture}.relinked", directory / f"composer-{fixture}"]
            for path in paths:
                if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
                    raise ValueError(f"missing emitted input: {path}")
                hashes[str(path.relative_to(build))] = digest(path)
            if not filecmp.cmp(paths[-2], paths[-1], shallow=False):
                raise ValueError(f"binary equivalence mismatch: {mode}/{fixture}")
    return {"files": len(hashes), "sha256": hashes}


if __name__ == "__main__":
    try:
        print(json.dumps(verify(Path(sys.argv[1])), sort_keys=True))
    except (IndexError, OSError, ValueError) as exc:
        print(f"emitted proof HOLD: {exc}", file=sys.stderr)
        raise SystemExit(86)
