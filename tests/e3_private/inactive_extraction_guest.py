"""Inactive, test-only E3 extraction guest; selected bundle must pin this file."""

from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import re
import sys


J_FILES = {
    "tests/g3c4/f_loader_outer.py": "73066210494ef0cf3f5bd824520fc70e73545f4176a6ae3eedabbe16d5a97360",
    "tests/g3c4/f_artifact_attest.py": "5aac55665ad5a1767c54591ec3530667cced310f07948257521f34ed43999e78",
    "tests/g3c4/f_image_loader_collect.py": "8b92aabe7fb742b81dc5e1d8fcd57bf92b9f5b3ad702920e9c826ce8faafd101",
}
RECIPE_SHA = "215346061c87c6cf2c3475fdd7f2cea4f88e3f30239bc1b8a987d079f683761d"


def ordinary(path: Path) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        if not path.is_file() or before.st_nlink != 1 or before.st_size > 1024 * 1024:
            raise ValueError("guest source is not selected ordinary input")
        raw = os.read(fd, before.st_size + 1)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if len(raw) != before.st_size or (before.st_dev, before.st_ino, before.st_mode,
        before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_dev, after.st_ino, after.st_mode, after.st_size,
        after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("guest selected source changed during read")
    return raw


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(expected_pins_sha256: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_pins_sha256):
        raise ValueError("externally fixed material pins required")
    for relative, expected in J_FILES.items():
        if sha256(ordinary(Path("/bundle/h-source") / relative)).hexdigest() != expected:
            raise ValueError("reviewed J guest dependency changed")
    if sha256(ordinary(Path("/bundle/e3_recipe.py"))).hexdigest() != RECIPE_SHA:
        raise ValueError("reviewed E3 recipe changed")
    sys.path.insert(0, "/bundle/h-source")
    from tests.g3c4 import f_loader_outer as h
    from tests.g3c4 import f_artifact_attest as attest
    recipe = load(Path("/bundle/e3_recipe.py"), "selected_e3_recipe")
    pins_raw = attest.ordinary(Path("/bundle/e3-material-pins.json"))
    if sha256(pins_raw).hexdigest() != expected_pins_sha256:
        raise ValueError("material pin table changed")
    pins = json.loads(pins_raw)
    if type(pins) is not dict or set(pins) != recipe.MATERIAL_NAMES:
        raise ValueError("E3 material table incomplete")
    material = {name: attest.ordinary(Path("/bundle/e3-material") / name)
                for name in pins}
    recipe.prepare(material, pins)
    Path("/out/phases").mkdir(mode=0o700)
    for mode in ("normal", "sanitize"):
        receipt = h.capture(list(recipe.guest_command(mode)),
                            Path("/out/phases") / mode)
        if receipt["returncode"] != 0 or not receipt["quiescent"] or receipt["failure"] or \
                attest.ordinary(Path("/out/phases") / mode / "stderr.raw"):
            raise ValueError(f"E3 {mode} extraction failed; raw phase retained")
    evidence = Path("/out/evidence")
    evidence.mkdir(mode=0o700)
    attest.exclusive_bytes(evidence / "extraction-guest.json", h.canonical({
        "format": "e3-sealed-fixture-extraction-guest-v1",
        "modes": ["normal", "sanitize"],
        "material_pins_sha256": expected_pins_sha256,
        "status": "EXTRACTION_ONLY",
    }))


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--material-pins-sha256":
        raise SystemExit("exact E3 material pin argument required")
    run(sys.argv[2])
