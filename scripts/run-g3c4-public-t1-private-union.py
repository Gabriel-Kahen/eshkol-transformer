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
import tempfile


ROOT = Path(__file__).resolve().parent.parent
PREFLIGHT = ROOT / "scripts/preflight-g3c4-public-t1-private-union.py"
SPEC = importlib.util.spec_from_file_location("private_union_source", PREFLIGHT)
assert SPEC and SPEC.loader
source = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(source)

# A ROOT-reviewed decision may set this in a later source successor. The
# decision pins the previously reviewed source basis, never its own successor.
ACCEPTED_DECISION_SHA256 = ""
FORMAT = "g3c4-private-union-launch-v2"
IMAGE = "sha256:f31d1db76958339e6ebd2a2f667052cdb85aeb5229914ffb10ac4fcdc6db22e6"
ROOT_REVIEW = "8069ed6dbbe4a01ec4f0ff1e13eea94d116b3da4073bb3d8dca76863f545d60d"
BASE_HEAD = "93844697dca03199d96e3ab91a27e5d140c0f6ca"
SOURCE_BASIS_HEAD = "2e8ba2b2be06e6a217eaf1d972c5150c43c03c51"
SOURCE_BASIS_TREE = "5debf4e5608f6b754a89380cd7fcdab283078ae6"
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


def ordinary_with_stat(path: Path) -> tuple[bytes, os.stat_result]:
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
        closed = os.fstat(fd)
        after = path.lstat()
        identity = lambda info: (info.st_dev, info.st_ino, info.st_mode,
                                 info.st_size, info.st_uid, info.st_gid,
                                 info.st_mtime_ns, info.st_ctime_ns)
        require(identity(closed) == identity(before) == identity(after),
                "decision or pin changed during read")
        return raw, closed
    finally:
        os.close(fd)


def ordinary(path: Path) -> bytes:
    return ordinary_with_stat(path)[0]


def runner_projection() -> str:
    """Hash runner bytes with only the embedded approval value erased."""
    raw = ordinary(Path(__file__).resolve())
    pattern = rb'(?m)^ACCEPTED_DECISION_SHA256 = "[0-9a-f]*"$'
    require(len(re.findall(pattern, raw)) == 1,
            "runner approval slot is not unique")
    return sha(re.sub(pattern, b'ACCEPTED_DECISION_SHA256 = ""', raw))


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
    raw, info = ordinary_with_stat(Path(row["path"]))
    require((sha(raw), len(raw), stat.S_IMODE(info.st_mode), info.st_uid, info.st_gid) ==
            (row["sha256"], row["size"], row["mode"], row["uid"], row["gid"]),
            f"selected pin drift: {row['role']}")


def owned_directory(directory: Path) -> None:
    info = directory.lstat()
    require(directory.is_absolute() and stat.S_ISDIR(info.st_mode) and
            not stat.S_ISLNK(info.st_mode) and info.st_uid == os.geteuid() and
            stat.S_IMODE(info.st_mode) == 0o700,
            "staging directory must be owned and private")


def stage_file(raw: bytes, target: Path, mode: int) -> None:
    require(not target.exists() and mode & 0o7000 == 0,
            "staged input exists or has special permission bits")
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                 os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
        os.fchmod(fd, mode)
        os.fsync(fd)
    finally:
        os.close(fd)


def stage_inputs(rows: dict[str, dict], directory: Path) -> tuple[dict[str, dict], Path]:
    """Copy authenticated inputs into an owned tree; no child uses live paths."""
    owned_directory(directory)
    pins = directory / "pins"
    staged: dict[str, dict] = {}
    for role, row in rows.items():
        raw, info = ordinary_with_stat(Path(row["path"]))
        require((sha(raw), len(raw), stat.S_IMODE(info.st_mode), info.st_uid,
                 info.st_gid) ==
                (row["sha256"], row["size"], row["mode"], row["uid"], row["gid"]),
                f"selected pin drift while staging: {role}")
        target = pins / role / Path(row["path"]).name
        stage_file(raw, target, row["mode"])
        staged[role] = {**row, "path": str(target), "uid": os.geteuid(),
                        "gid": os.getegid()}
        verify_pin(staged[role])
    source_root = directory / "source"
    for name in sorted(source.source_names(ROOT)):
        raw = source.ordinary(ROOT, name)
        require(raw == source.git(ROOT, "show", f"{source.BASE}:{name}"),
                f"reviewed source changed while staging: {name}")
        mode = source.git(ROOT, "ls-tree", source.BASE, "--", name).split(None, 1)
        require(mode and mode[0] in (b"100644", b"100755"),
                f"reviewed source mode drifted while staging: {name}")
        stage_file(raw, source_root / name, int(mode[0][-3:], 8))
    return staged, source_root


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
    require(set(data) == {"format", "source_basis_head", "source_basis_tree", "base_head",
                          "runner_projection_sha256", "preflight_sha256",
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
    require((data["source_basis_head"], data["source_basis_tree"]) ==
            (SOURCE_BASIS_HEAD, SOURCE_BASIS_TREE) and
            source.git(ROOT, "rev-parse", f"{SOURCE_BASIS_HEAD}^{{tree}}").decode().strip() ==
            SOURCE_BASIS_TREE and
            source.git(ROOT, "merge-base", "--is-ancestor", SOURCE_BASIS_HEAD,
                       "HEAD") == b"",
            "reviewed source basis mismatch")
    require(data["runner_projection_sha256"] == runner_projection() and
            data["preflight_sha256"] == sha(ordinary(PREFLIGHT)),
            "reviewed runner/preflight source projection mismatch")
    recheck_inputs(rows, check_source=check_source, check_pin=check_pin,
                   read_source=read_source)
    return data, rows


def aot_argv(rows: dict[str, dict], lane: str, mode: str,
             directory: Path, source_root: Path) -> tuple[str, ...]:
    require((lane, mode) in ARCHIVES and directory.is_absolute() and
            source_root.is_absolute(),
            "unknown private lane or mode")
    archive = Path(rows[ARCHIVES[(lane, mode)]]["path"])
    stage_root = source_root.parent
    owned_directory(stage_root)
    pins_root = stage_root / "pins"
    tool = Path(rows["eshkol_run"]["path"])
    require(source_root == stage_root / "source" and source_root.is_dir() and
            tool.parent == pins_root / "eshkol_run" and
            archive.parent == pins_root / ARCHIVES[(lane, mode)],
            "AOT inputs must be owned staged paths")
    require(archive.name == f"libg3c4_private_{lane}_{mode}.a" and
            archive.name != "libeshkol_transformer_g3g_manual.a",
            "private archive identity drifted")
    return (str(tool), "--strict-types", "--optimize", "0",
            "--no-stdlib", "-I", str(source_root / source.INCLUDES[0]),
            "-I", str(source_root / source.INCLUDES[1]),
            "-I", str(source_root / source.INCLUDES[2]),
            "-I", str(source_root / source.INCLUDES[3]),
            "-I", str(source_root / "lib"), "-I", str(source_root / "native"),
            "-L", str(archive.parent), "--lib", archive.stem[3:],
            str(source_root / FIXTURES[lane]), "-o", str(directory / "caller"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("decision", type=Path)
    args = parser.parse_args()
    try:
        raw = ordinary(args.decision)
        decision, rows = admit(raw, ACCEPTED_DECISION_SHA256)
        with tempfile.TemporaryDirectory(prefix="g3c4-private-owned-") as temporary:
            stage_inputs(rows, Path(temporary))
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
