#!/usr/bin/env python3
"""Build and capture the noninstalled E3 first-warmup AOT detector tuple.

Run only after source review and immutable toolchain/runner admission. An abort
cannot close LSan; the separate complete-call and native controls must close.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tests.e3_private.parse_first_eval_detection import SELECTORS, verify  # noqa: E402

PREFIX = "retention_first_eval_mutant_package_"
PACKAGE = "e3_retention_package.o"
ARCHIVE = "libeshkol_transformer_e3_retention_mutant.a"
MANIFESTS = {
    "defined_symbols": "global-defined",
    "native_objects": "native-objects", "native_source_closure": "native-source-closure",
    "public_exports": "package-exports",
    "public_strings": "public-strings", "source_closure": "source-closure",
    "undefined_symbols": "undefined", "sanitized_instrumentation_undefined_symbols":
    "sanitizer-undefined", "sanitized_noninstrumentation_undefined_symbols":
    "noninstrumentation-undefined",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def capture(out: Path, name: str, argv: list[str], env: dict[str, str],
            cwd: Path, deadline: int) -> dict[str, object]:
    """Own raw streams, wait status, group teardown and per-child resource use."""
    require(deadline > 0 and all(isinstance(arg, str) for arg in argv),
            "invalid capture command or deadline")
    stem = out / name
    started = time.monotonic()
    with stem.with_suffix(".stdout").open("xb") as stdout, \
         stem.with_suffix(".stderr").open("xb") as stderr:
        child = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stdout,
                                 stderr=stderr, start_new_session=True,
                                 preexec_fn=lambda: resource.setrlimit(
                                     resource.RLIMIT_CORE, (0, 0)))
        timed_out = False
        while True:
            pid, wait, usage = os.wait4(child.pid, os.WNOHANG)
            if pid:
                break
            if time.monotonic() - started >= deadline:
                timed_out = True
                os.killpg(child.pid, signal.SIGTERM)
                stop = time.monotonic() + 5
                while time.monotonic() < stop:
                    pid, wait, usage = os.wait4(child.pid, os.WNOHANG)
                    if pid:
                        break
                    time.sleep(0.05)
                if not pid:
                    os.killpg(child.pid, signal.SIGKILL)
                    _, wait, usage = os.wait4(child.pid, 0)
                break
            time.sleep(0.05)
        code = os.waitstatus_to_exitcode(wait)
        child.returncode = code
        try:
            os.killpg(child.pid, 0)
        except ProcessLookupError:
            survivors = False
        else:
            survivors = True
            os.killpg(child.pid, signal.SIGKILL)
    result = {
        "argv": argv, "cwd": str(cwd), "env": env, "deadline_seconds": deadline,
        "elapsed_seconds": time.monotonic() - started, "wait_status": wait,
        "exit_status": code if code >= 0 else None,
        "signal": os.WTERMSIG(wait) if os.WIFSIGNALED(wait) else None,
        "returncode": code, "timed_out": timed_out, "survivors": survivors,
        "teardown": "SIGKILL process group" if survivors else
                    ("timeout TERM/KILL" if timed_out else "none"),
        "max_rss_kib": usage.ru_maxrss, "user_seconds": usage.ru_utime,
        "system_seconds": usage.ru_stime,
        "stdout_sha256": digest(stem.with_suffix(".stdout")),
        "stderr_sha256": digest(stem.with_suffix(".stderr")),
    }
    stem.with_suffix(".json").write_text(json.dumps(result, sort_keys=True,
                                                     indent=2) + "\n")
    return result


def good(stage: dict[str, object], label: str) -> None:
    require(stage["returncode"] == 0 and not stage["timed_out"] and
            not stage["survivors"], f"{label} did not close successfully")


def same(left: Path, right: Path) -> None:
    require(left.read_bytes() == right.read_bytes(), f"bytes differ: {left} {right}")


def check_package(artifact: Path, sanitized: bool) -> None:
    expected = ROOT / "tests/e3_private"
    observed = artifact / (PACKAGE + ".evidence")
    for manifest, evidence in MANIFESTS.items():
        if manifest.startswith("sanitized_") and not sanitized:
            continue
        if manifest == "undefined_symbols" and sanitized:
            continue
        same(expected / (PREFIX + manifest + ".txt"), observed / (evidence + ".txt"))
    members = subprocess.check_output(["ar", "t", str(artifact / ARCHIVE)])
    require(members == (expected / (PREFIX + "archive_members.txt")).read_bytes(),
            "archive member inventory differs")
    if sanitized:
        union = sorted(set((expected / (PREFIX +
                           "sanitized_instrumentation_undefined_symbols.txt")).read_text().splitlines()) |
                       set((expected / (PREFIX +
                           "sanitized_noninstrumentation_undefined_symbols.txt")).read_text().splitlines()))
        require((observed / "undefined.txt").read_text().splitlines() == union,
                "sanitized undefined-symbol union differs")
    symbols = (observed / "readelf-symbols.txt").read_text()
    require(" et_e3_test_first_eval_corrupt_v1\n" in symbols and
            any("LOCAL" in line and line.endswith("et_e3_test_first_eval_corrupt_v1")
                for line in symbols.splitlines()), "mutant helper is not private")


def tracked_source_hashes() -> dict[str, str]:
    require(not subprocess.check_output(["git", "status", "--porcelain"],
                                        cwd=ROOT), "runner requires a clean committed union")
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).split(b"\0")
    return {name.decode(): digest(ROOT / name.decode()) for name in files if name}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="fresh absolute evidence directory")
    args = parser.parse_args()
    out = args.output
    require(out.is_absolute() and not out.exists() and not out.is_symlink() and
            out.parent.is_dir() and not out.parent.is_symlink() and
            not out.is_relative_to(ROOT),
            "output must be a fresh absolute directory with a real parent")
    source_hashes = tracked_source_hashes()
    out.mkdir(mode=0o700)
    (out / "source-sha256.json").write_text(json.dumps(source_hashes,
                                                        sort_keys=True, indent=2) + "\n")
    (out / "source-commit.txt").write_bytes(subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT))
    (out / "source-tree.txt").write_bytes(subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT))
    provenance = Path(os.environ.get("ESHKOL_BUILD_DIR", ROOT / ".deps/eshkol-build")) / \
        "eshkol-transformer-provenance.tsv"
    values = dict(line.split("\t", 1) for line in provenance.read_text().splitlines())
    cxx = values["cxx_path"]
    build = provenance.parent
    source = Path(os.environ.get("ESHKOL_SOURCE_DIR", ROOT / ".deps/eshkol-src"))
    require(Path(cxx).is_absolute() and Path(cxx).is_file() and
            (build / "eshkol-run").is_file() and source.is_dir(),
            "selected compiler/source/toolchain is unavailable")
    toolchain_files = [provenance, Path(values["cc_path"]), Path(cxx),
                       build / "eshkol-run",
                       build / "libeshkol-runtime.a", build / "libeshkol-static.a"]
    require(all(path.is_file() for path in toolchain_files),
            "selected runtime archives are unavailable")
    (out / "toolchain-sha256.json").write_text(json.dumps(
        {str(path): digest(path) for path in toolchain_files},
        sort_keys=True, indent=2) + "\n")
    env = {key: os.environ[key] for key in ("PATH", "HOME") if key in os.environ}
    env.update(LANG="C", LC_ALL="C", ESHKOL_SOURCE_DIR=str(source),
               ESHKOL_BUILD_DIR=str(build), ESHKOL_JIT_CACHE="0",
               ESHKOL_LIB_DIR=str(ROOT / "lib"))
    build_timeout = int(os.environ.get("E3_COMPILER_TIMEOUT_SECONDS", "900"))
    runtime_timeout = int(os.environ.get("E3_FIRST_EVAL_TIMEOUT_SECONDS", "3600"))
    require(build_timeout > 0 and runtime_timeout > 0, "timeouts must be positive")
    artifacts = {}
    for mode in ("a", "b", "sanitized"):
        artifact = out / ("private-" + mode)
        stage = capture(out, "build-" + mode,
                        ["/usr/bin/bash", str(ROOT / "scripts/build-e3-retention-first-eval-mutant.sh"),
                         str(artifact), "sanitize" if mode == "sanitized" else "normal"],
                        env, ROOT, build_timeout)
        good(stage, "package " + mode)
        check_package(artifact, mode == "sanitized")
        artifacts[mode] = artifact
    same(artifacts["a"] / PACKAGE, artifacts["b"] / PACKAGE)
    same(artifacts["a"] / ARCHIVE, artifacts["b"] / ARCHIVE)
    evidence_a = artifacts["a"] / (PACKAGE + ".evidence")
    evidence_b = artifacts["b"] / (PACKAGE + ".evidence")
    same(evidence_a / "private.d", evidence_b / "private.d")
    same(evidence_a / "readelf-symbols.txt", evidence_b / "readelf-symbols.txt")
    require({p.name for p in (evidence_a / "native-depfiles").iterdir()} ==
            {p.name for p in (evidence_b / "native-depfiles").iterdir()},
            "native depfile inventory differs")
    for depfile in (evidence_a / "native-depfiles").iterdir():
        same(depfile, evidence_b / "native-depfiles" / depfile.name)
    corpus = out / "corpus"
    good(capture(out, "corpus", [sys.executable, "-m", "tests.e3_reference.corpus",
                                 "--output", str(corpus)], env, ROOT, build_timeout), "corpus")
    fixture = corpus / "packed-single"
    require(fixture.is_dir(), "retention corpus is absent")
    cc_args = [cxx, "-std=c++17", "-Wall", "-Wextra", "-Werror",
               "-fstack-protector-all", "-fPIC", "-fvisibility=hidden",
               "-fno-common", "-I", str(source / "lib/core"), "-I",
               str(source / "inc"), "-c", str(ROOT / "tests/e3_private/retention_arena.cpp")]
    for mode in ("a", "sanitized"):
        obj = out / ("arena-" + mode + ".o")
        flags = ["-O1", "-g", "-fsanitize=address,undefined",
                 "-fno-omit-frame-pointer"] if mode == "sanitized" else []
        good(capture(out, "arena-" + mode, cc_args + flags + ["-o", str(obj)],
                     env, ROOT, build_timeout), "arena " + mode)
    wrapper = out / "cxx-sanitized"
    wrapper.write_text("#!/usr/bin/env bash\nexec " + shlex.quote(cxx) +
                       ' -fsanitize=address,undefined -fno-omit-frame-pointer "$@"\n')
    wrapper.chmod(0o500)
    compiled = {}
    linked = {}
    for mode in ("a", "b", "sanitized"):
        link = out / ("link-" + mode)
        link.mkdir()
        good(capture(out, "extract-" + mode,
                     ["ar", "x", str(artifacts[mode] / ARCHIVE), PACKAGE],
                     env, link, build_timeout), "archive extraction")
        same(link / PACKAGE, artifacts[mode] / PACKAGE)
        arena = out / ("arena-sanitized.o" if mode == "sanitized" else "arena-a.o")
        archive = link / "libeshkol_transformer_e3_retention_mutant_test.a"
        linked[mode] = capture(out, "link-" + mode,
                               ["ar", "rcsD", str(archive), str(link / PACKAGE),
                                str(arena)], env, ROOT, build_timeout)
        good(linked[mode], "archive link")
        compiler_env = env | {
            "XDG_CACHE_HOME": str(out / ("cache-" + mode)),
            "ESHKOL_CXX_COMPILER": str(wrapper if mode == "sanitized" else cxx),
        }
        executable = out / ("runtime-" + mode)
        compile_stage = capture(out, "compile-" + mode,
                                [str(build / "eshkol-run"), "--strict-types",
                                 "--no-stdlib", "-O", "2", "-L", str(link),
                                 "--lib", "eshkol_transformer_e3_retention_mutant_test",
                                 str(ROOT / "tests/e3_private/retention_first_eval_mutant_runtime.esk"),
                                 "-o", str(executable)], compiler_env, ROOT,
                                build_timeout)
        good(compile_stage, "AOT compile/link " + mode)
        compiled[mode] = compile_stage
        require(executable.is_file(), "AOT executable absent")
        nm_stage = capture(out, "nm-" + mode, ["nm", "-u", str(executable)],
                           env, ROOT, build_timeout)
        good(nm_stage, "AOT unresolved inventory")
        require(b"et_e3_retention_region_used_v1" not in
                (out / ("nm-" + mode + ".stdout")).read_bytes(),
                "arena observer remained unresolved")
    same(out / "link-a/libeshkol_transformer_e3_retention_mutant_test.a",
         out / "link-b/libeshkol_transformer_e3_retention_mutant_test.a")
    same(out / "runtime-a", out / "runtime-b")
    hashes = {p.relative_to(out).as_posix(): digest(p) for p in out.rglob("*")
              if p.is_file() and not p.name.endswith((".stdout", ".stderr", ".json"))}
    (out / "artifacts-sha256.json").write_text(json.dumps(hashes,
                                                           sort_keys=True, indent=2) + "\n")
    for selector, gradient in SELECTORS.items():
        for horizon in (1024, 8192):
            for profile in ("normal-a", "normal-b", "poisoned-a", "poisoned-b",
                            "sanitized"):
                mode = "sanitized" if profile == "sanitized" else profile[-1]
                runtime_env = env | {
                    "ESHKOL_ARENA_POISON": "1" if profile.startswith("poisoned-") or
                                              profile == "sanitized" else "0",
                    "ASAN_OPTIONS": "detect_leaks=1:halt_on_error=1",
                    "UBSAN_OPTIONS": "halt_on_error=1",
                }
                name = f"{horizon}-{gradient}-{selector}-{profile}"
                stage = capture(out, name,
                                [str(out / ("runtime-" + mode)), str(fixture),
                                 str(horizon), gradient, selector], runtime_env,
                                ROOT, runtime_timeout)
                verify(selector=selector, gradient=gradient, horizon=horizon,
                       package="retention-first-eval-mutant",
                       compile_status=compiled[mode]["returncode"],
                       link_status=linked[mode]["returncode"],
                       runtime_status=stage["returncode"],
                       runtime_signal=stage["signal"], timed_out=stage["timed_out"],
                       survivors=stage["survivors"],
                       stdout=(out / (name + ".stdout")).read_bytes(),
                       stderr=(out / (name + ".stderr")).read_bytes())
    sealed = sorted(p for p in out.rglob("*") if p.is_file())
    (out / "SHA256SUMS").write_text("".join(
        f"{digest(path)}  {path.relative_to(out).as_posix()}\n" for path in sealed))
    print("E3 first-warmup AOT detector accepted: " + str(out))


if __name__ == "__main__":
    main()
