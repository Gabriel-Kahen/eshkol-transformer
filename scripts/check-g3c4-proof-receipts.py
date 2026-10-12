#!/usr/bin/env python3
"""Fail-closed offline validator for the terminal-close nested command graph."""
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
import re
import sys

_trace_spec = importlib.util.spec_from_file_location(
    "g3c4_native_trace_receipt", Path(__file__).with_name("g3c4_native_trace_receipt.py"))
g3c4_native_trace_receipt = importlib.util.module_from_spec(_trace_spec)
_trace_spec.loader.exec_module(g3c4_native_trace_receipt)

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


def python_site(name: str, fragment: str) -> str:
    project = Path(__file__).resolve().parents[1]
    source = project / ("tests/q0" if name == "test_python_isolation.py" else "scripts")
    matches = [number for number, line in enumerate((source / name).read_text().splitlines(), 1)
               if fragment in line]
    require(len(matches) == 1, f"ambiguous Python child site: {name} {fragment}")
    return f"{name}:{matches[0]}"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


Q0_SITE = python_site("test_python_isolation.py", "tracked = subprocess.run(")
CLONE_CC_SITE = python_site("check-g3c4-p2g2-terminal-ids-clone.py",
                            "return subprocess.run(command,")
CLONE_GIT_SITE = python_site("check-g3c4-p2g2-terminal-ids-clone.py",
                             "old.write_bytes(subprocess.run(")


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
        '"${resolved_cc}"': 'clang-21',
        '"${resolved_cxx}"': 'clang++-21',
        '"${command}"': 'llvm-config-21',
        '"${binary}"': 'eshkol-run',
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
    # These common.sh sites are on every supported terminal-close path. The
    # diagnostic-only e1b allowlist branch is outside this proof invocation.
    for number, line in enumerate((source / "common.sh").read_text().splitlines(), 1):
        if "common_probe" not in line or "common_probe()" in line:
            continue
        if "cmp -s" in line or "comm -3" in line:
            continue
        programs = re.findall(r"\bcommon_probe\s+(\S+)", line)
        if programs:
            calls[f"common.sh:{number}"] = Counter(map(expected_program, programs))
    calls[Q0_SITE] = Counter({"git": 1})
    calls[CLONE_CC_SITE] = Counter({"clang-21": 4})
    calls[CLONE_GIT_SITE] = Counter({"git": 2})
    for name, fragment, count in (
        ("check-g3c4-p2g2-t1-prefix-coordinator.py", "subprocess.run([sys.executable,", 1),
        ("check-g3c4-p2g2-protected-scope.py", "subprocess.run([sys.executable,", 1),
        ("check-g3c4-p2g2-ownership.py", "subprocess.run([sys.executable,", 3),
        ("check-g3c4-p2g2-carrier-bridge.py", "subprocess.run(", 1),
        ("check-g3c4-p2g2-prefix-commit.py", "subprocess.run([sys.executable,", 1),
        ("check-g3c4-p2g2-first-frame.py", "subprocess.run([sys.executable,", 2),
    ):
        calls[python_site(name, fragment)] = Counter({"@python3": count})
    calls[python_site("check-g3c4-p2g2-terminal.py",
                      "return subprocess.run(command,")] = Counter({"clang-21": 8})
    calls[python_site("check-g3c4-p2g2-terminal.py",
                      "old.write_bytes(subprocess.run(")] = Counter({"git": 4})
    return calls


def source_sites() -> set[str]:
    return set(source_calls())


def verify(root: Path, expected_sites: set[str] | dict[str, Counter] | None = None,
           native_log: Path | None = None) -> dict:
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
    native_receipts = []
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
        require(all(type(meta[field]) is int for field in
                    ("wall_ns", "start_monotonic_ns", "end_monotonic_ns")) and
                meta["wall_ns"] >= 0 and
                meta["end_monotonic_ns"] - meta["start_monotonic_ns"] == meta["wall_ns"],
                f"bad wall interval: {path}")
        cwd = (path / "cwd").read_bytes()
        require(cwd.startswith(b"/") and cwd.endswith(b"\n"), f"bad cwd: {path}")
        argv_raw = (path / "argv.nul").read_bytes()
        require(argv_raw.endswith(b"\0") and len(argv_raw) > 1,
                f"bad argv: {path}")
        require(hashlib.sha256(argv_raw).hexdigest() == meta["argv_sha256"],
                f"argv hash mismatch: {path}")
        native_receipts.append((argv_raw, meta["start_monotonic_ns"],
                                meta["end_monotonic_ns"]))
        environment_hash = path / "environment.sha256"
        require(environment_hash.is_file() and not environment_hash.is_symlink() and
                re.fullmatch(r"[0-9a-f]{64}\n", environment_hash.read_text()),
                f"bad environment hash: {path}")
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
            if program == '*':
                found = sum(actual.values())
            elif program == '@python3':
                found = sum(number for name, number in actual.items()
                            if re.fullmatch(r'python3(?:\.[0-9]+)?', name))
            else:
                found = actual[program]
            require(found >= count, f"missing command receipt: {site} {program}")
    result = {"receipts": len(paths), "scripts": sorted(scripts),
              "source_sites": len(expected_sites)}
    if native_log is not None:
        result.update(g3c4_native_trace_receipt.verify(native_log, native_receipts))
    return result


if __name__ == "__main__":
    try:
        require(len(sys.argv) in (2, 4) and
                (len(sys.argv) == 2 or sys.argv[2] == "--native-log"),
                "usage: check-g3c4-proof-receipts.py ROOT [--native-log LOG]")
        native_log = Path(sys.argv[3]) if len(sys.argv) == 4 else None
        print(json.dumps(verify(Path(sys.argv[1]), native_log=native_log), sort_keys=True))
    except (IndexError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"proof receipts HOLD: {exc}", file=sys.stderr)
        raise SystemExit(86)
