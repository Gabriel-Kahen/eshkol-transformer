#!/usr/bin/env python3
"""Decision-gated private one-root AOT route. No launch seal is admitted yet."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent
PREFLIGHT = ROOT / "scripts/preflight-g3c4-public-t1-private-union.py"
SPEC = importlib.util.spec_from_file_location("private_union_source", PREFLIGHT)
assert SPEC and SPEC.loader
source = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(source)

# A ROOT-reviewed launch decision must set this in a separately reviewed source
# successor. No caller-supplied hash or environment override can release it.
ACCEPTED_DECISION_SHA256 = ""
FORMAT = "g3c4-private-union-launch-v1"
IMAGE = "sha256:f31d1db76958339e6ebd2a2f667052cdb85aeb5229914ffb10ac4fcdc6db22e6"
ROOT_REVIEW = "8069ed6dbbe4a01ec4f0ff1e13eea94d116b3da4073bb3d8dca76863f545d60d"
BASE_HEAD = "93844697dca03199d96e3ab91a27e5d140c0f6ca"
REQUIRED_ROLES = frozenset((
    "image_attestation", "source_admission", "slot_receipt",
    "compiler_source", "compiler_lock",
    "eshkol_run", "cc", "cxx", "cxx_sanitize", "runtime_archive",
    "static_archive", "generated_headers", "loader_closure", "tool_inventory",
    "runtime_inventory", "provider_inventory", "symbol_inventory",
    "history_normal_archive", "history_sanitize_archive",
    "t1_normal_archive", "t1_sanitize_archive",
    "t1_native_normal", "t1_native_sanitize",
))
LIMITS = frozenset(("wall_seconds", "cpu_seconds", "memory_bytes", "swap_bytes",
                    "pids", "file_bytes", "output_bytes", "tmp_bytes",
                    "min_free_bytes"))
FIXTURES = {"history": source.HISTORY, "t1": source.T1}
ARCHIVES = {("history", "normal"): "history_normal_archive",
            ("history", "sanitize"): "history_sanitize_archive",
            ("t1", "normal"): "t1_normal_archive",
            ("t1", "sanitize"): "t1_sanitize_archive"}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def ordinary(path: Path) -> bytes:
    require(path.is_absolute() and path.parent.resolve(strict=True) == path.parent,
            "decision or pin has a symlinked parent")
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
            "decision or pin is not a unique ordinary file")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        require((opened.st_dev, opened.st_ino, opened.st_mode, opened.st_size) ==
                (before.st_dev, before.st_ino, before.st_mode, before.st_size),
                "decision or pin identity changed before read")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read()
        after = path.lstat()
        require((after.st_dev, after.st_ino, after.st_mode, after.st_size,
                 after.st_mtime_ns, after.st_ctime_ns) ==
                (before.st_dev, before.st_ino, before.st_mode, before.st_size,
                 before.st_mtime_ns, before.st_ctime_ns),
                "decision or pin changed during read")
        return raw
    finally:
        os.close(fd)


def canonical(raw: bytes) -> dict:
    data = json.loads(raw)
    require(type(data) is dict and
            raw == (json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n").encode(),
            "launch decision is not canonical JSON")
    return data


def pin_rows(data: dict) -> dict[str, dict]:
    rows = data.get("pins")
    require(type(rows) is list and rows, "missing selected pin table")
    by_role: dict[str, dict] = {}
    seen_paths: set[str] = set()
    for row in rows:
        require(type(row) is dict and
                set(row) == {"role", "path", "sha256", "size", "mode", "uid", "gid"},
                "malformed pin row")
        role, path = row["role"], row["path"]
        require(type(role) is str and re.fullmatch(r"[a-z][a-z0-9_]*", role) and
                type(path) is str and Path(path).is_absolute() and
                role not in by_role and path not in seen_paths,
                "duplicate or malformed selected pin")
        require(type(row["sha256"]) is str and
                re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) and
                all(type(row[key]) is int and row[key] >= 0
                    for key in ("size", "mode", "uid", "gid")),
                "malformed selected pin metadata")
        by_role[role] = row
        seen_paths.add(path)
    require(REQUIRED_ROLES <= by_role.keys(), "missing required selected pin role")
    return by_role


def verify_pin(row: dict) -> None:
    path = Path(row["path"])
    raw = ordinary(path)
    info = path.lstat()
    require((sha(raw), len(raw), stat.S_IMODE(info.st_mode), info.st_uid, info.st_gid) ==
            (row["sha256"], row["size"], row["mode"], row["uid"], row["gid"]),
            f"selected pin drift: {row['role']}")


def recheck_inputs(rows: dict[str, dict], *,
                   check_source=source.source_topology,
                   check_pin=verify_pin,
                   read_source=source.ordinary) -> None:
    check_source(ROOT)
    for lane, fixture in FIXTURES.items():
        source.one_root(read_source(ROOT, fixture), lane)
    for row in rows.values():
        check_pin(row)


def admit(raw: bytes, approved_sha256: str, *,
          check_source=source.source_topology,
          check_pin=verify_pin,
          read_source=source.ordinary) -> tuple[dict, dict[str, dict]]:
    require(re.fullmatch(r"[0-9a-f]{64}", approved_sha256) is not None and
            sha(raw) == approved_sha256, "no ROOT-approved exact launch decision")
    data = canonical(raw)
    require(set(data) == {"format", "source_head", "source_tree", "base_head",
                          "root_review_sha256", "image", "pins", "limits",
                          "slot_receipt_sha256"} and
            data["format"] == FORMAT and data["image"] == IMAGE and
            data["base_head"] == BASE_HEAD and
            data["root_review_sha256"] == ROOT_REVIEW and
            type(data["slot_receipt_sha256"]) is str and
            re.fullmatch(r"[0-9a-f]{64}", data["slot_receipt_sha256"]),
            "launch decision schema or accepted source pins drifted")
    limits = data["limits"]
    require(type(limits) is dict and set(limits) == LIMITS and
            all(type(value) is int and value > 0 for value in limits.values()),
            "missing package-phase resource bounds")
    rows = pin_rows(data)
    require(rows["slot_receipt"]["sha256"] == data["slot_receipt_sha256"],
            "owned heavy slot receipt drifted")
    head, tree = source.git(ROOT, "rev-parse", "HEAD", "HEAD^{tree}").decode().splitlines()
    require((data["source_head"], data["source_tree"]) == (head, tree),
            "source commit/tree mismatch")
    recheck_inputs(rows, check_source=check_source, check_pin=check_pin,
                   read_source=read_source)
    return data, rows


def aot_argv(rows: dict[str, dict], lane: str, mode: str,
             directory: Path) -> tuple[str, ...]:
    require((lane, mode) in ARCHIVES and directory.is_absolute(),
            "unknown private lane or mode")
    archive = Path(rows[ARCHIVES[(lane, mode)]]["path"])
    require(archive.name == f"libg3c4_private_{lane}_{mode}.a" and
            archive.name != "libeshkol_transformer_g3g_manual.a",
            "private archive identity drifted")
    return (rows["eshkol_run"]["path"], "--strict-types", "--optimize", "0",
            "--no-stdlib", "-I", str(ROOT / source.INCLUDES[0]),
            "-I", str(ROOT / source.INCLUDES[1]),
            "-I", str(ROOT / source.INCLUDES[2]),
            "-I", str(ROOT / source.INCLUDES[3]),
            "-I", str(ROOT / "lib"), "-I", str(ROOT / "native"),
            "-L", str(archive.parent), "--lib", archive.stem[3:],
            str(ROOT / FIXTURES[lane]), "-o", str(directory / "caller"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("decision", type=Path)
    args = parser.parse_args()
    try:
        raw = ordinary(args.decision)
        admit(raw, ACCEPTED_DECISION_SHA256)
    except (OSError, ValueError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired, UnicodeDecodeError,
            json.JSONDecodeError) as exc:
        print(f"HOLD before compiler/Docker child: {exc}", file=sys.stderr)
        return 2
    # The AOT argv contract is available for a reviewed launch successor. This
    # source revision has no child route: native closure and source-to-use
    # ownership still need an independently accepted implementation.
    print("HOLD: accepted pins require a reviewed child/receipt route", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
