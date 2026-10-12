#!/usr/bin/env python3
"""Fail-closed offline validator for the terminal-close nested command graph."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import re
import sys

SCRIPTS = (
    "test-g3c4-p2g2-terminal-close.sh",
    "test-g3c4-p2g2-t1-prefix-coordinator.sh",
    "test-g3c4-p2g1-pending.sh",
    "test-g3c4-p2g2-protected-scope.sh",
    "test-g3c4-p2g2-carrier-bridge.sh",
    "test-g3c4-p2g2-prefix-commit.sh",
    "test-g3c4-p2g2-first-frame.sh",
    "test-g3c4-p2g2-terminal-ids-clone.sh",
    "test-g3c4-p2g2-terminal.sh",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate receipt field: {key}")
        result[key] = value
    return result


def expected_program(token: str) -> str:
    return {
        '"$cc"': 'clang-21', '"$cxx"': 'clang++-21',
        '"$runner"': 'eshkol-run', '"$eshkol_runner"': 'eshkol-run',
        '"${runtime[@]}"': 'env',
    }.get(token, '*' if token.startswith('"$') else token)


def source_calls() -> dict[str, Counter]:
    source = Path(__file__).resolve().parents[1] / "scripts"
    calls = {}
    for name in SCRIPTS:
        lines = (source / name).read_text().splitlines()
        for number, line in enumerate(lines, 1):
            programs = re.findall(r"\bproof_command\s+(\S+)", line)
            if programs:
                calls[f"{name}:{number}"] = Counter(map(expected_program, programs))
    return calls


def source_sites() -> set[str]:
    return set(source_calls())


def verify(root: Path, expected_sites: set[str] | dict[str, Counter] | None = None) -> dict:
    require(root.is_dir() and not root.is_symlink(), "receipt root missing or symlink")
    if expected_sites is None:
        expected_sites = source_calls()
    elif isinstance(expected_sites, set):
        expected_sites = {site: Counter({'*': 1}) for site in expected_sites}
    paths = sorted(root.glob("command.*"))
    require(len(paths) >= len(SCRIPTS), "missing command receipts")
    scripts = set()
    invoked = set()
    observed_sites = set()
    observed_programs = {}
    for path in paths:
        require(path.is_dir() and not path.is_symlink(), f"bad receipt directory: {path}")
        meta_path = path / "receipt.json"
        require(meta_path.is_file() and not meta_path.is_symlink(), f"missing receipt: {path}")
        meta = json.loads(meta_path.read_text(), object_pairs_hook=unique_object)
        fields = ("exit", "start_monotonic_ns", "end_monotonic_ns", "wall_ns",
                  "stdout_bytes", "stderr_bytes", "stdout_sha256",
                  "stderr_sha256", "argv_sha256", "script", "site")
        require(all(field in meta for field in fields), f"missing field: {path}")
        require(type(meta["exit"]) is int and 0 <= meta["exit"] <= 255,
                f"bad numeric exit: {path}")
        require(meta["exit"] != 86, f"capture failure: {path}")
        require(type(meta["wall_ns"]) is int and meta["wall_ns"] >= 0 and
                meta["end_monotonic_ns"] - meta["start_monotonic_ns"] == meta["wall_ns"],
                f"bad wall interval: {path}")
        cwd = (path / "cwd").read_bytes()
        require(cwd.startswith(b"/") and cwd.endswith(b"\n"), f"bad cwd: {path}")
        argv_raw = (path / "argv.nul").read_bytes()
        require(argv_raw.endswith(b"\0") and len(argv_raw) > 1,
                f"bad argv: {path}")
        require(hashlib.sha256(argv_raw).hexdigest() == meta["argv_sha256"],
                f"argv hash mismatch: {path}")
        argv = [x.decode("utf-8", "surrogateescape") for x in argv_raw[:-1].split(b"\0")]
        for stream in ("stdout", "stderr"):
            data_path = path / f"{stream}.raw"
            require(data_path.is_file() and not data_path.is_symlink(),
                    f"missing raw stream: {path}")
            data = data_path.read_bytes()
            require(len(data) <= 16 * 1024 * 1024 and
                    len(data) == meta[f"{stream}_bytes"] and
                    hashlib.sha256(data).hexdigest() == meta[f"{stream}_sha256"],
                    f"stream mismatch: {path}/{stream}")
        scripts.add(meta["script"])
        observed_sites.add(meta["site"])
        observed_programs.setdefault(meta["site"], Counter())[Path(argv[0]).name] += 1
        if len(argv) >= 2 and Path(argv[0]).name == "bash":
            invoked.add(Path(argv[1]).name)
    require(set(SCRIPTS) <= scripts, "nested script command receipts missing")
    require(set(SCRIPTS[1:]) <= invoked, "nested predecessor invocation missing")
    require(set(expected_sites) <= observed_sites,
            f"unexecuted command sites: {sorted(set(expected_sites) - observed_sites)}")
    for site, required in expected_sites.items():
        actual = observed_programs[site]
        for program, count in required.items():
            found = sum(actual.values()) if program == '*' else actual[program]
            require(found >= count, f"missing command receipt: {site} {program}")
    return {"receipts": len(paths), "scripts": sorted(scripts),
            "source_sites": len(expected_sites)}


if __name__ == "__main__":
    try:
        print(json.dumps(verify(Path(sys.argv[1])), sort_keys=True))
    except (IndexError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"proof receipts HOLD: {exc}", file=sys.stderr)
        raise SystemExit(86)
