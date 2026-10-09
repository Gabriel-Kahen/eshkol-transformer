"""Test-only C4 native capture in a pinned guest, then closed-evidence Q0 comparison.

This module never starts a container and its host route does not attest guest closure.
The outer admitted gate must supply an immutable, physically closed guest tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import stat
import subprocess
import sys
import tempfile

from tests.g3c4 import c4_native_transcript as transcript
from tests.g3c4 import c4_toolchain_profile as profile

ROOT = Path(__file__).resolve().parents[2]
BASE_COMMIT = "6b18976d1f3f900a333c9e99122a1a603664b37d"
BASE_TREE = "2d238af456dc5657a21806fcadda12dfae637cdf"
FIXTURE_SHA = transcript.FIXTURE_SHA256
TOOL_PINS_SHA = "18764bbd7e634c0c01c3ca8f0337fa34f73d92b0631bffb23e25c3a1ef850cc7"
RUNTIME_PINS_SHA = "94a7c39142c213681da20ba640996d14504d4c697dcc88c925af35577dedb53c"
LOCK_SHA = "b798822b18535405092b201248d8e37682d5ddb9555b6080ee32131659bfbea5"
Q0_LOCK_SHA = "fa99f800a188b8aded75702fbbd7d74f2ccd813e10bb8b23c1b469f70c24601a"
CLOSURE = ROOT / "native/g3c4_c4_split_capture_source_closure.txt"
SOURCES = (
    "tests/g3c4/c4_native_capture.c",
    "src/eshkol_transformer/m3_i64_integration.c",
    "native/t1_i64_shell.c",
    "native/n3k_primitives_provider.c",
    "native/g3c4_primitives_provider.c",
    "native/g3s_sampling_provider.c",
    "native/g3n_primitives_provider.c",
    "native/n2_primitives_provider.c",
    "native/a2_kv_cache.c",
)
CHANNELS = ("normal/first", "normal/repeat", "sanitize/first")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def bounded_read(path: Path, limit: int) -> bytes:
    require(path.stat().st_size <= limit, "oversized capture artifact")
    raw = path.read_bytes()
    require(len(raw) <= limit, "capture artifact grew during read")
    return raw


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode()


def write_new(path: Path, raw: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(raw)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def git(*args: str) -> str:
    return subprocess.run(("/usr/bin/git", "--no-optional-locks", "-C",
                           str(ROOT), *args),
                          check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=30).stdout.decode().strip()


def source_state(expected_head: str, expected_tree: str) -> dict:
    require(len(expected_head) == len(expected_tree) == 40 and
            all(c in "0123456789abcdef" for c in expected_head + expected_tree),
            "source admission pin format")
    require(git("rev-parse", f"{BASE_COMMIT}^{{tree}}") == BASE_TREE,
            "accepted C4 base tree drift")
    git("merge-base", "--is-ancestor", BASE_COMMIT, expected_head)
    require(git("rev-parse", "HEAD") == expected_head and
            git("rev-parse", "HEAD^{tree}") == expected_tree and
            git("status", "--porcelain=v1", "--untracked-files=all") == "",
            "C4 exact clean source mismatch")
    rows = CLOSURE.read_text().splitlines()
    require(len(rows) == 217 and len(set(rows)) == len(rows),
            "split source closure membership")
    for member in rows:
        require(member and not member.startswith("/") and
                ".." not in Path(member).parts and (ROOT / member).is_file(),
                "split source closure member")
    return {"commit": expected_head, "tree": expected_tree,
            "closure": [[member, file_digest(ROOT / member)] for member in rows]}


def _pin_rows(path: Path, expected_sha: str, kind: str) -> list[tuple[str, str, str]]:
    raw = path.read_bytes()
    require(digest(raw) == expected_sha, f"{kind} pin manifest drift")
    rows = []
    for line in raw.decode("ascii").splitlines():
        fields = line.split("\t")
        if kind == "tool":
            require(len(fields) == 3, "tool pin fields")
            source, target, sha = fields
        else:
            require(len(fields) == 2, "runtime pin fields")
            sha, source = fields
            target = source
        require(source.startswith("/usr/") and target.startswith("/usr/") and
                len(sha) == 64 and all(c in "0123456789abcdef" for c in sha),
                f"{kind} pin path/digest")
        rows.append((source, target, sha))
    require(len(rows) == (36 if kind == "tool" else 33) and
            len(set(source for source, _, _ in rows)) == len(rows),
            f"{kind} pin count/duplicates")
    return rows


def pinned_files(path: Path, expected_sha: str, kind: str) -> list[tuple[str, str, str]]:
    result = []
    for source, target, sha in _pin_rows(path, expected_sha, kind):
        actual = Path(source).resolve(strict=True)
        require((kind != "tool" or str(actual) == target) and actual.is_file() and
                file_digest(actual) == sha, f"{kind} input drift: {source}")
        result.append((source, str(actual), sha))
    return result


def guest_state(fixture: Path, lock: Path, tools: Path, runtimes: Path,
                expected_head: str, expected_tree: str) -> dict:
    require(os.environ.get("PATH") == "/usr/bin:/bin" and
            os.environ.get("LC_ALL") == "C" and
            os.environ.get("TZ") == "UTC" and
            os.environ.get("PYTHONDONTWRITEBYTECODE") == "1",
            "guest fixed environment")
    require("ASAN_OPTIONS" not in os.environ and
            "UBSAN_OPTIONS" not in os.environ,
            "guest sanitizer environment must begin clean")
    require(platform.system() == "Linux" and platform.machine() == "x86_64",
            "unsupported guest platform")
    release = Path("/etc/os-release").read_text()
    require('ID=ubuntu\n' in release and 'VERSION_ID="22.04"\n' in release,
            "unsupported guest OS")
    require(os.environ.get("ESHKOL_ALLOW_UNSUPPORTED_HOST", "0") == "0",
            "unsupported-host bypass forbidden")
    require(fixture.is_absolute() and lock.is_absolute() and
            tools.is_absolute() and runtimes.is_absolute(), "absolute guest inputs")
    raw_lock, lock_identity = profile.ordinary_external_lock(lock, ROOT)
    require(file_digest(fixture) == FIXTURE_SHA and
            digest(raw_lock) == LOCK_SHA, "fixture/candidate lock drift")
    require(sys.version_info[:2] == (3, 10) and
            Path(sys.executable).resolve() == Path("/usr/bin/python3.10"),
            "guest Python 3.10 identity")
    return {"source": source_state(expected_head, expected_tree),
            "fixture": FIXTURE_SHA,
            "candidate_lock": lock_identity,
            "os_release": digest(release.encode()),
            "python": file_digest(Path(sys.executable).resolve()),
            "tools": [list(row) for row in pinned_files(tools, TOOL_PINS_SHA,
                                                         "tool")],
            "runtimes": [list(row) for row in pinned_files(
                runtimes, RUNTIME_PINS_SHA, "runtime")]}


def stage(directory: Path, name: str, argv: list[str], timeout: int = 120) -> bytes:
    """Run an exact command using the accepted bounded owned-child capture."""
    require(all(isinstance(arg, str) and arg for arg in argv) and
            Path(argv[0]).is_absolute(), "stage argv")
    wrapper = directory / (name + ".sh")
    write_new(wrapper, ("#!/usr/bin/bash\nexec " +
                        " ".join(shlex.quote(arg) for arg in argv) + "\n").encode())
    wrapper.chmod(0o500)
    write_new(directory / (name + ".argv.json"), canonical(argv))
    return transcript.run_native(wrapper, b"", directory / name,
                                 timeout_seconds=timeout)


def build_mode(output: Path, mode: str) -> Path:
    require(mode in ("normal", "sanitize"), "capture build mode")
    directory = output / mode
    objects = directory / "objects"
    objects.mkdir(parents=True, mode=0o700)
    flags = (["-O2"] if mode == "normal" else
             ["-O1", "-g", "-fsanitize=address,undefined",
              "-fno-omit-frame-pointer"])
    compiled = []
    for source in SOURCES:
        obj = objects / (Path(source).stem + ".o")
        stage(directory, "compile-" + Path(source).stem,
              ["/usr/bin/clang-21", "-std=c11", "-Wall", "-Wextra",
               "-Werror", "-Wpedantic", "-ffp-contract=off",
               "-fexcess-precision=standard", "-fno-fast-math",
               "-fstack-protector-all", *flags,
               "-I", str(ROOT / "include"), "-I", str(ROOT / "native"),
               "-I", str(ROOT / "src"),
               "-I", str(ROOT / "src/eshkol_transformer"),
               "-DET_I64_TENSOR_TESTING", "-DET_A2_KV_CACHE_TESTING",
               "-DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE",
               "-c", str(ROOT / source), "-o", str(obj)])
        require(obj.is_file(), "C4 object missing")
        compiled.append(str(obj))
    archive = directory / "libc4_native_capture.a"
    stage(directory, "archive", ["/usr/bin/ar", "rcsD", str(archive),
                                  *compiled])
    binary = directory / "c4-native-capture"
    stage(directory, "link", ["/usr/bin/clang-21", *flags, *compiled,
                               "-Wl,--wrap=et_a2_kv_cache_transaction_abort_v1",
                               "-lm", "-o", str(binary)])
    require(binary.is_file() and os.access(binary, os.X_OK),
            "C4 native executable missing")
    for name, argv in (
        ("archive-defined.symbols", ["/usr/bin/nm", "-g", "--defined-only",
                                     "--format=posix", str(archive)]),
        ("archive-undefined.symbols", ["/usr/bin/nm", "-u", "--format=posix",
                                       str(archive)]),
        ("defined.symbols", ["/usr/bin/nm", "-g", "--defined-only",
                             "--format=posix", str(binary)]),
        ("undefined.symbols", ["/usr/bin/nm", "-u", "--format=posix",
                               str(binary)]),
        ("elf.symbols", ["/usr/bin/readelf", "--wide", "--symbols",
                         str(binary)])):
        write_new(directory / name,
                  stage(directory, "inspect-" + name.replace(".", "-"), argv))
    return binary


def inventory(root: Path) -> dict[str, str]:
    rows = {}
    directories = set()
    for path in root.rglob("*"):
        kind = path.lstat().st_mode
        require(not stat.S_ISLNK(kind) and
                (stat.S_ISREG(kind) or stat.S_ISDIR(kind)),
                "nonordinary guest evidence member")
        if stat.S_ISREG(kind):
            name = path.relative_to(root).as_posix()
            if name != "GUEST-INVENTORY.json":
                rows[name] = file_digest(path)
        else:
            directories.add(path.relative_to(root).as_posix())
    require(directories == {"tmp", "normal", "normal/objects",
                            "sanitize", "sanitize/objects"},
            "guest evidence directory census")
    return dict(sorted(rows.items()))


def expected_members() -> set[str]:
    result = {"INPUTS.pre.json", "INPUTS.post.json", "REQUEST.txt"}
    inspections = ("archive-defined.symbols", "archive-undefined.symbols",
                   "defined.symbols", "undefined.symbols", "elf.symbols")
    for mode in ("normal", "sanitize"):
        stages = (["compile-" + Path(source).stem for source in SOURCES] +
                  ["archive", "link"] +
                  ["inspect-" + name.replace(".", "-")
                   for name in inspections])
        for stage_name in stages:
            for suffix in (".sh", ".argv.json", ".stdout", ".stderr",
                           ".status.json"):
                result.add(f"{mode}/{stage_name}{suffix}")
        for source in SOURCES:
            result.add(f"{mode}/objects/{Path(source).stem}.o")
        result.update(f"{mode}/{name}" for name in inspections)
        result.add(f"{mode}/libc4_native_capture.a")
        result.add(f"{mode}/c4-native-capture")
        for run in (("first", "repeat") if mode == "normal" else ("first",)):
            for suffix in (".stdout", ".stderr", ".status.json", ".json"):
                result.add(f"{mode}/{run}{suffix}")
    return result


def require_receipt(prefix: Path, binary: str, stdout_limit: int) -> None:
    status_raw = bounded_read(prefix.with_suffix(".status.json"), 16384)
    status = json.loads(status_raw)
    require(type(status) is dict and canonical(status) == status_raw and
            set(status) == {"binary", "sha256_pre", "sha256_post",
                            "returncode", "leader_identity", "uid",
                            "elapsed_seconds", "timeout_seconds",
                            "wait_status", "quiescent", "timed_out",
                            "overflow", "stdout_seen", "stderr_seen",
                            "teardown", "failure", "stdout_bytes",
                            "stderr_bytes"}, "child receipt schema")
    leader = status["leader_identity"]
    raw = bounded_read(prefix.with_suffix(".stdout"), stdout_limit)
    stderr = bounded_read(prefix.with_suffix(".stderr"), transcript.MAX_STDERR_BYTES)
    require(status["binary"] == binary and
            type(leader) is list and len(leader) == 5 and
            all(type(value) is int and value > 0 for value in leader[:4]) and
            leader[0] == leader[1] == leader[2] and
            type(leader[4]) is str and type(status["uid"]) is int and
            status["uid"] >= 0 and type(status["elapsed_seconds"]) in
            (int, float) and 0 <= status["elapsed_seconds"] <= 130 and
            type(status["timeout_seconds"]) is int and
            1 <= status["timeout_seconds"] <= 120 and
            status["returncode"] == 0 and status["wait_status"] == 0 and
            status["quiescent"] is True and status["failure"] == "" and
            status["teardown"] == [] and status["timed_out"] is False and
            status["overflow"] == "" and
            status["stdout_seen"] == status["stdout_bytes"] == len(raw) and
            status["stderr_seen"] == status["stderr_bytes"] == len(stderr) == 0 and
            len(raw) <= stdout_limit, "child receipt outcome")
    require(status["sha256_pre"] == status["sha256_post"],
            "child executable hash drift")


def guest(fixture: Path, lock: Path, tools: Path, runtimes: Path,
          output: Path, expected_head: str, expected_tree: str) -> None:
    require(output.is_absolute() and not output.exists(), "fresh guest output")
    output.mkdir(mode=0o700)
    original_tmp = os.environ.get("TMPDIR")
    original_tempdir = tempfile.tempdir
    tmp = output / "tmp"
    try:
        tmp.mkdir(mode=0o700)
        os.environ["TMPDIR"] = str(tmp)
        tempfile.tempdir = str(tmp)
        before = guest_state(fixture, lock, tools, runtimes,
                             expected_head, expected_tree)
        before["guest_output"] = str(output)
        write_new(output / "INPUTS.pre.json", canonical(before))
        frozen = transcript.load_fixture(fixture)
        request = transcript.make_input(frozen)
        write_new(output / "REQUEST.txt", request)
        normal = build_mode(output, "normal")
        safe = build_mode(output, "sanitize")
        for mode, binary, name in (("normal", normal, "first"),
                                   ("normal", normal, "repeat"),
                                   ("sanitize", safe, "first")):
            directory = output / mode
            if mode == "sanitize":
                os.environ["ASAN_OPTIONS"] = "detect_leaks=1:halt_on_error=1"
                os.environ["UBSAN_OPTIONS"] = "halt_on_error=1"
            try:
                raw = transcript.run_native(binary, request,
                                            directory / name)
            finally:
                os.environ.pop("ASAN_OPTIONS", None)
                os.environ.pop("UBSAN_OPTIONS", None)
            write_new(directory / (name + ".json"),
                      transcript.encode_observation(
                          transcript.parse(raw, frozen)))
        streams = [bounded_read(output / (name + ".stdout"),
                                transcript.MAX_RAW_BYTES)
                   for name in CHANNELS]
        require(streams[0] == streams[1] == streams[2],
                "native normal/repeat/sanitizer raw drift")
        after = guest_state(fixture, lock, tools, runtimes,
                            expected_head, expected_tree)
        after["guest_output"] = str(output)
        write_new(output / "INPUTS.post.json", canonical(after))
        require(before == after, "guest source/tool/runtime drift")
        members = inventory(output)
        require(set(members) == expected_members(), "guest evidence member census")
        write_new(output / "GUEST-INVENTORY.json", canonical(members))
    except BaseException as error:
        try:
            write_new(output / "GUEST-FAILURE.json",
                      canonical({"error": type(error).__name__,
                                 "detail": str(error)}))
        except BaseException as secondary:
            print(f"guest failure receipt unavailable: {secondary}",
                  file=sys.stderr)
        raise
    finally:
        if original_tmp is None:
            os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = original_tmp
        tempfile.tempdir = original_tempdir


def verify_closed_tree(root: Path, expected_head: str,
                       expected_tree: str) -> dict:
    """Check durable bytes; physical closure remains an outer admission fact."""
    inventory_raw = bounded_read(root / "GUEST-INVENTORY.json", 131072)
    sealed = json.loads(inventory_raw)
    require(type(sealed) is dict and canonical(sealed) ==
            inventory_raw and
            sealed == inventory(root) and set(sealed) == expected_members(),
            "guest durable inventory mismatch")
    before = bounded_read(root / "INPUTS.pre.json", 131072)
    require(before == bounded_read(root / "INPUTS.post.json", 131072),
            "guest input identity drift")
    state = json.loads(before)
    require(state["source"]["commit"] == expected_head and
            state["source"]["tree"] == expected_tree and
            state["fixture"] == FIXTURE_SHA and
            state["candidate_lock"]["sha256"] == LOCK_SHA,
            "guest input pin mismatch")
    guest_output = state["guest_output"]
    require(type(guest_output) is str and guest_output.startswith("/"),
            "guest output identity")
    inspections = ("archive-defined.symbols", "archive-undefined.symbols",
                   "defined.symbols", "undefined.symbols", "elf.symbols")
    for mode in ("normal", "sanitize"):
        for stage_name in (["compile-" + Path(source).stem for source in SOURCES] +
                           ["archive", "link"] +
                           ["inspect-" + name.replace(".", "-")
                            for name in inspections]):
            prefix = root / mode / stage_name
            original = str(Path(guest_output) / mode / (stage_name + ".sh"))
            require_receipt(prefix, original, transcript.MAX_RAW_BYTES)
            args_raw = bounded_read(prefix.with_suffix(".argv.json"), 65536)
            args = json.loads(args_raw)
            require(type(args) is list and all(type(arg) is str and arg
                                               for arg in args) and
                    args_raw == canonical(args) and
                    bounded_read(prefix.with_suffix(".sh"), 65536) ==
                    ("#!/usr/bin/bash\nexec " +
                     " ".join(shlex.quote(arg) for arg in args) +
                     "\n").encode(), "stage argv/wrapper drift")
            require(file_digest(prefix.with_suffix(".sh")) ==
                    json.loads(bounded_read(prefix.with_suffix(".status.json"),
                                            16384))
                    ["sha256_pre"], "stage wrapper hash drift")
    for channel in CHANNELS:
        prefix = root / channel
        original = str(Path(guest_output) / channel.split("/")[0] /
                       "c4-native-capture")
        require_receipt(prefix, original, transcript.MAX_RAW_BYTES)
        status = json.loads(bounded_read(prefix.with_suffix(".status.json"),
                                         16384))
        binary = root / channel.split("/")[0] / "c4-native-capture"
        sha = file_digest(binary)
        require(status["sha256_pre"] == sha == status["sha256_post"],
                "guest binary drift")
    return state


def q0_state() -> dict:
    require(os.environ.get("ATEN_CPU_CAPABILITY") == "default" and
            os.environ.get("MKL_CBWR") == "COMPATIBLE",
            "host Q0 environment")
    import torch
    require(sys.version_info[:3] == (3, 14, 6) and
            torch.__version__ == "2.13.0+cpu", "host Q0 runtime")
    executable = Path(sys.executable).resolve(strict=True)
    module = Path(torch.__file__).resolve(strict=True)
    core = Path(torch._C.__file__).resolve(strict=True)
    lock = ROOT / "tests/q0/requirements-oracle.lock"
    require(file_digest(lock) == Q0_LOCK_SHA, "host Q0 lock drift")
    return {"executable": str(executable),
            "executable_sha256": file_digest(executable),
            "torch_module": str(module),
            "torch_module_sha256": file_digest(module),
            "torch_core": str(core),
            "torch_core_sha256": file_digest(core),
            "q0_lock_sha256": Q0_LOCK_SHA}


def host(fixture: Path, guest_root: Path, output: Path,
         expected_head: str, expected_tree: str) -> None:
    require(fixture.is_absolute() and guest_root.is_absolute() and
            output.is_absolute() and not output.exists(), "host path admission")
    output.mkdir(mode=0o700)
    try:
        require(file_digest(fixture) == FIXTURE_SHA, "host fixture drift")
        state = verify_closed_tree(guest_root, expected_head, expected_tree)
        require(state["source"] == source_state(expected_head, expected_tree),
                "guest/host C4 source drift")
        frozen = transcript.load_fixture(fixture)
        raws = [bounded_read(guest_root / (name + ".stdout"),
                             transcript.MAX_RAW_BYTES)
                for name in CHANNELS]
        require(raws[0] == raws[1] == raws[2], "native raw repeat/sanitizer drift")
        from tests.g3c4 import c4_math_oracle as oracle
        q0_before = q0_state()
        oracle.require_q0()
        oracle.check_source_pins()
        expected = oracle.decode(fixture.read_bytes())
        measured = {}
        for channel, raw in zip(CHANNELS, raws):
            actual = transcript.encode_observation(transcript.parse(raw, frozen))
            require(actual == (guest_root / (channel + ".json")).read_bytes(),
                    "guest/host canonical drift")
            measured[channel] = oracle.compare_observation(
                expected, oracle.decode(actual), FIXTURE_SHA)
        require(file_digest(fixture) == FIXTURE_SHA and
                state["source"] == source_state(expected_head, expected_tree) and
                q0_before == q0_state() and
                verify_closed_tree(guest_root, expected_head, expected_tree) == state,
                "host post-admission drift")
        write_new(output / "HOST-COMPARISON.json",
                  canonical({"guest_inventory_sha256": file_digest(
                      guest_root / "GUEST-INVENTORY.json"),
                             "source_head": expected_head,
                             "source_tree": expected_tree,
                             "fixture_sha256": FIXTURE_SHA,
                             "raw_sha256": {name: digest(raw)
                                            for name, raw in zip(CHANNELS, raws)},
                             "q0_record": q0_before,
                             "measured": measured}))
    except BaseException as error:
        try:
            write_new(output / "HOST-FAILURE.json",
                      canonical({"error": type(error).__name__,
                                 "detail": str(error)}))
        except BaseException as secondary:
            print(f"host failure receipt unavailable: {secondary}",
                  file=sys.stderr)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    routes = parser.add_subparsers(dest="route", required=True)
    guest_parser = routes.add_parser("guest")
    guest_parser.add_argument("--fixture", required=True, type=Path)
    guest_parser.add_argument("--candidate-lock", required=True, type=Path)
    guest_parser.add_argument("--tool-pins", required=True, type=Path)
    guest_parser.add_argument("--runtime-pins", required=True, type=Path)
    guest_parser.add_argument("--source-head", required=True)
    guest_parser.add_argument("--source-tree", required=True)
    guest_parser.add_argument("--output", required=True, type=Path)
    host_parser = routes.add_parser("host")
    host_parser.add_argument("--fixture", required=True, type=Path)
    host_parser.add_argument("--closed-guest", required=True, type=Path)
    host_parser.add_argument("--source-head", required=True)
    host_parser.add_argument("--source-tree", required=True)
    host_parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.route == "guest":
        guest(args.fixture, args.candidate_lock, args.tool_pins,
              args.runtime_pins, args.output,
              args.source_head, args.source_tree)
    else:
        host(args.fixture, args.closed_guest, args.output,
             args.source_head, args.source_tree)


if __name__ == "__main__":
    main()
