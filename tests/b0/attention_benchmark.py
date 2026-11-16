#!/usr/bin/env python3
"""Versioned, direct-native A2 CPU-f32 attention benchmark (development tooling)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from . import benchmark_format as codec
    from . import run_benchmark as common
else:
    import benchmark_format as codec
    import run_benchmark as common

ROOT = Path(__file__).resolve().parents[2]
DEFINITION = "eshkol-attention-benchmark-definition"
REPORT = "eshkol-attention-benchmark-report"
SHAPES = [[1,2,2,1,1,1], [1,2,2,2,2,1], [1,2,2,1,2,1],
          [1,2,2,1,2,2], [1,2,2,2,2,2], [1,4,2,1,1,2],
          [1,4,2,3,3,2], [2,4,2,2,3,4], [2,4,2,3,3,2], [2,4,2,1,3,2]]
SOURCES = ["benchmarks/attention_v1.c", "benchmarks/measure_linux.c",
           "native/a2_attention_provider.c", "native/kernel_abi.c",
           "include/eshkol_transformer/a2_attention_abi.h",
           "include/eshkol_transformer/kernel_abi.h", "tests/b0/attention_benchmark.py",
           "tests/b0/benchmark_format.py", "tests/b0/run_benchmark.py"]
SUPPORT_SOURCES = ["scripts/verify-attention-supported.sh", "scripts/common.sh",
                   "toolchain/eshkol.lock"]
FLAGS = ["-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-Wpedantic", "-ffp-contract=off"]
CONTRACT = {
    "id": "a2-attention-v1", "backend": "a2.cpu-f32.serial", "device": "cpu", "dtype": "f32",
    "shapes": SHAPES, "operations": ["forward", "backward"],
    "input_recipe": "dyadic-positions-selective-mask-v1", "deterministic": True,
    "measurement": {"clock": "clock_monotonic", "dispatch_scope": "k1-dispatch-batch",
                    "process_scope": "fresh-process-including-setup-and-warmup",
                    "memory_source": "linux_wait4_ru_maxrss", "memory_unit": "kibibyte",
                    "completion_source": "linux_pidfd_poll"},
    "warmup": 10, "iterations": 1000, "repetitions": 3, "timeout_ms": 30000,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise codec.BenchmarkFormatError(message)


def keys(value: object, names: set[str], where: str) -> None:
    codec._keys(codec._object(value, where), names, where)


def validate_definition(value: object) -> dict:
    keys(value, set(CONTRACT), "definition")
    for name in set(CONTRACT) - {"warmup", "iterations", "repetitions", "timeout_ms"}:
        # Canonical equality also distinguishes bool from integer values.
        require(codec.canonical_json(value[name]) == codec.canonical_json(CONTRACT[name]),
                f"unsupported definition {name}")
    for name, lower, upper in [("warmup",0,100000), ("iterations",1,1000000),
                               ("repetitions",1,100), ("timeout_ms",1,3600000)]:
        codec._integer(value[name], name, lower, upper)
    return value


def encode_definition(value: dict) -> bytes:
    return codec._encode(DEFINITION, value, validate_definition)


def load_definition(path: Path | bytes) -> dict:
    return codec._load(path, DEFINITION, validate_definition)


def summary(samples: list[dict], iterations: int) -> dict:
    elapsed = sorted(s["dispatch_elapsed_ns"] for s in samples)
    return {"dispatch_ns_min": elapsed[0], "dispatch_ns_median": elapsed[(len(elapsed)-1)//2],
            "dispatch_ns_max": elapsed[-1], "process_peak_rss_kib": max(s["process_peak_rss_kib"] for s in samples),
            "dispatches_per_second": codec.decimal_rate(iterations*len(samples), sum(elapsed))}


def parse_support_tsv(raw: str, name: str) -> dict[str, str]:
    require(len(raw.encode("utf-8")) <= 65536, f"{name} is too large")
    result: dict[str, str] = {}
    for line in raw.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        require(len(fields) == 2 and all(fields) and fields[0] not in result,
                f"invalid {name} TSV")
        result[fields[0]] = fields[1]
    return result


def validate_support_evidence(stable: dict, volatile: dict) -> None:
    evidence = stable["support_evidence"]
    keys(evidence, {"lock_tsv", "provenance_tsv", "provenance_sha256"},
         "support evidence")
    lock_raw = codec._string(evidence["lock_tsv"], "lock TSV", nonempty=True)
    provenance_raw = codec._string(evidence["provenance_tsv"], "provenance TSV", nonempty=True)
    require(hashlib.sha256(lock_raw.encode()).hexdigest() ==
            stable["sources"]["toolchain/eshkol.lock"], "lock source hash mismatch")
    for path in SOURCES + SUPPORT_SOURCES:
        require(common.sha256_file(ROOT/path) == stable["sources"][path],
                f"benchmark source changed: {path}")
    require(hashlib.sha256(provenance_raw.encode()).hexdigest() ==
            codec._sha256(evidence["provenance_sha256"], "provenance digest"),
            "provenance hash mismatch")
    lock = parse_support_tsv(lock_raw, "lock")
    provenance = parse_support_tsv(provenance_raw, "provenance")
    for name, expected in {"supported_os": "ubuntu-22.04", "supported_arch": "x86_64",
                           "llvm_version": "21.1.8", "clang_version": "21.1.8",
                           "supported_cc": "clang-21"}.items():
        require(lock.get(name) == expected, f"unsupported lock {name}")
    for name in ("eshkol_repository", "eshkol_commit", "eshkol_version"):
        require(provenance.get(name) == lock.get(name) and bool(lock.get(name)),
                f"toolchain {name} mismatch")
    for name, lock_name in (("llvm_version", "llvm_version"),
                            ("cc_version", "clang_version"),
                            ("cxx_version", "clang_version")):
        require(provenance.get(name) == lock[lock_name], f"toolchain {name} mismatch")
    require(provenance.get("cc_path") == stable["compiler"]["path"],
            "compiler path/provenance mismatch")
    require(Path(provenance["cc_path"]).is_absolute(), "compiler path is not absolute")
    require(re.search(r"\bversion 21\.1\.8\b", stable["compiler"]["version"].splitlines()[0]) is not None,
            "compiler version/provenance mismatch")
    codec._sha256(provenance.get("eshkol_binary_sha256"), "Eshkol binary digest")
    host = volatile["support_evidence"]
    keys(host, {"os_id", "os_version_id", "architecture"}, "support host")
    require(host["os_id"] == "ubuntu" and host["os_version_id"] == "22.04",
            "unsupported report host")
    require(host["architecture"] == "x86_64" and
            host["architecture"] == volatile["environment"]["architecture"],
            "unsupported report architecture")


def validate_report(value: object, *, version: int = 1) -> dict:
    require(version in (1, 2), "unsupported attention report version")
    keys(value, {"stable", "volatile"}, "report")
    stable, volatile = value["stable"], value["volatile"]
    stable_keys = {"definition", "definition_sha256", "project", "sources", "compiler", "binaries"}
    if version == 2:
        stable_keys.add("support_evidence")
    keys(stable, stable_keys, "stable")
    definition = validate_definition(stable["definition"])
    expected_digest = hashlib.sha256(encode_definition(definition)).hexdigest()
    require(stable["definition_sha256"] == expected_digest, "definition digest mismatch")
    keys(stable["project"], {"commit", "dirty"}, "project")
    require(re.fullmatch(r"[0-9a-f]{40}", codec._string(stable["project"]["commit"], "commit")) is not None,
            "invalid project commit")
    require(type(stable["project"]["dirty"]) is bool, "invalid dirty flag")
    keys(stable["sources"], set(SOURCES + (SUPPORT_SOURCES if version == 2 else [])), "sources")
    keys(stable["binaries"], {"target", "helper"}, "binaries")
    for digest in [*stable["sources"].values(), *stable["binaries"].values()]:
        codec._sha256(digest, "source/binary digest")
    compiler = stable["compiler"]
    keys(compiler, {"path", "sha256", "version", "flags"}, "compiler")
    for name in ["path", "version"]:
        codec._string(compiler[name], f"compiler {name}", nonempty=True)
    codec._sha256(compiler["sha256"], "compiler digest")
    require(compiler["flags"] == FLAGS, "unsupported compiler flags")
    volatile_keys = {"observed_at_utc", "support_status", "environment", "cases"}
    if version == 2:
        volatile_keys.add("support_evidence")
    keys(volatile, volatile_keys, "volatile")
    require(volatile["support_status"] == ("supported" if version == 2 else "compatibility-only"),
            "unsupported admission claim")
    timestamp = codec._string(volatile["observed_at_utc"], "observation time")
    require(codec._UTC_TIMESTAMP.fullmatch(timestamp) is not None, "invalid UTC timestamp")
    try:
        datetime.fromisoformat(timestamp[:-1] + "+00:00")
    except ValueError as error:
        raise codec.BenchmarkFormatError("invalid UTC timestamp") from error
    environment = volatile["environment"]
    keys(environment, {"architecture", "kernel", "cpu_model", "logical_cpus", "os_release"}, "environment")
    for name in set(environment) - {"logical_cpus"}:
        codec._string(environment[name], name, nonempty=True)
    codec._integer(environment["logical_cpus"], "logical_cpus", 1)
    cases = codec._array(volatile["cases"], "cases")
    require(len(cases) == len(SHAPES)*2, "missing or excess cases")
    for index, case in enumerate(cases):
        keys(case, {"shape", "operation", "output_sha256", "samples", "summary"}, "case")
        require(codec.canonical_json(case["shape"]) == codec.canonical_json(SHAPES[index//2]) and case["operation"] == definition["operations"][index%2],
                "case identity/order mismatch")
        codec._sha256(case["output_sha256"], "output digest")
        samples = codec._array(case["samples"], "samples")
        require(len(samples) == definition["repetitions"], "sample count mismatch")
        for repetition, sample in enumerate(samples):
            keys(sample, {"repetition", "dispatch_elapsed_ns", "process_elapsed_ns", "process_peak_rss_kib", "output_sha256"}, "sample")
            codec._integer(sample["repetition"], "repetition")
            require(sample["repetition"] == repetition, "sample order mismatch")
            require(sample["output_sha256"] == case["output_sha256"], "repetition output mismatch")
            for name in set(sample) - {"repetition", "output_sha256"}:
                codec._integer(sample[name], name, 1)
            require(sample["process_elapsed_ns"] >= sample["dispatch_elapsed_ns"], "dispatch exceeds process time")
        expected_summary = summary(samples, definition["iterations"])
        require(codec.canonical_json(case["summary"]) == codec.canonical_json(expected_summary), "summary mismatch")
    if version == 2:
        require(stable["project"]["dirty"] is False, "supported project tree is dirty")
        validate_support_evidence(stable, volatile)
    return value


def encode_report(value: dict, *, version: int = 1) -> bytes:
    return codec._encode(REPORT, value, lambda payload: validate_report(payload, version=version),
                         version=version)


def load_report(path: Path | bytes) -> dict:
    raw = path.read_bytes() if isinstance(path, Path) else path
    try:
        return codec._load(raw, REPORT, validate_report)
    except codec.BenchmarkFormatError as error:
        if str(error) != "unsupported benchmark version 2":
            raise
    return codec._load(raw, REPORT, lambda payload: validate_report(payload, version=2), version=2)


def native_output(raw: bytes, shape: list[int], operation: str) -> tuple[int, str]:
    try:
        value = json.loads(raw, object_pairs_hook=codec._pairs, parse_float=codec._forbid_float,
                           parse_constant=codec._forbid_constant)
    except (ValueError, UnicodeError) as error:
        raise codec.BenchmarkFormatError(f"malformed native output: {error}") from error
    keys(value, {"version", "provider", "dispatch_elapsed_ns", "output_bits"}, "native output")
    require(type(value["version"]) is int and value["version"] == 1 and value["provider"] == CONTRACT["backend"],
            "unexpected native protocol/provider")
    elapsed = codec._integer(value["dispatch_elapsed_ns"], "dispatch_elapsed_ns", 1)
    n,hq,hkv,tq,tk,d = shape
    count = n*hq*tq*d + (2*n*hkv*tk*d if operation == "backward" else 0)
    bits = value["output_bits"]
    require(isinstance(bits,str) and re.fullmatch(r"[0-9a-f]+", bits) is not None and len(bits) == count*8,
            "invalid output bits/shape")
    require(all((int(bits[i:i+8],16) & 0x7f800000) != 0x7f800000 for i in range(0,len(bits),8)),
            "nonfinite native output")
    # Each f32 word is eight hexadecimal digits, most-significant digit first;
    # the hashed witness is therefore independent of native byte order.
    require(any(int(bits[i:i+8],16) & 0x7fffffff for i in range(0,len(bits),8)), "all-zero native output")
    return elapsed, hashlib.sha256(bytes.fromhex(bits)).hexdigest()


def measure(helper: Path, target: Path, row: int, operation: str, definition: dict,
            directory: Path, label: str) -> tuple[dict, str]:
    out, err = directory / f"{label}.out", directory / f"{label}.err"
    argv = [str(helper), str(definition["timeout_ms"]), str(out), str(err), "--", str(target),
            str(row), operation, str(definition["warmup"]), str(definition["iterations"])]
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True,
                            timeout=definition["timeout_ms"]/1000 + 5, check=False)
    require(result.returncode == 0 and not result.stderr, "measurement helper failed: " + result.stderr)
    metrics = common.parse_metrics(result.stdout)
    require(not metrics["timed_out"] and metrics["exit_code"] == 0 and metrics["term_signal"] == 0,
            "native workload failed, signaled, or timed out")
    require(err.read_bytes() == b"", "native workload wrote stderr")
    elapsed, digest = native_output(out.read_bytes(), SHAPES[row], operation)
    return {"dispatch_elapsed_ns": elapsed, "process_elapsed_ns": metrics["elapsed_ns"],
            "process_peak_rss_kib": metrics["peak_rss_kib"]}, digest


def build(compiler: str, directory: Path) -> tuple[Path, Path, dict]:
    path = shutil.which(compiler)
    require(path is not None, "C compiler unavailable")
    compiler_path = Path(path).resolve()
    version = subprocess.run([str(compiler_path), "--version"], capture_output=True, text=True, check=True).stdout.strip()
    directory.mkdir(parents=True, exist_ok=True)
    target, helper = directory / "attention-v1", directory / "measure-linux-v1"
    subprocess.run([str(compiler_path), *FLAGS, "-I", str(ROOT/"include"),
                    *[str(ROOT/p) for p in ["benchmarks/attention_v1.c", "native/a2_attention_provider.c", "native/kernel_abi.c"]],
                    "-lm", "-o", str(target)], check=True)
    subprocess.run([str(compiler_path), *FLAGS, str(ROOT/"benchmarks/measure_linux.c"), "-o", str(helper)], check=True)
    return target, helper, {"path": str(compiler_path), "sha256": common.sha256_file(compiler_path),
                            "version": version, "flags": FLAGS}


def supported_probe(compiler: str) -> dict[str, str]:
    require("ESHKOL_ALLOW_UNSUPPORTED_HOST" not in os.environ,
            "supported attention measurement forbids host override")
    result = subprocess.run(["/usr/bin/bash", str(ROOT/"scripts/verify-attention-supported.sh"), compiler],
                            cwd=ROOT, capture_output=True, text=True, check=False)
    require(result.returncode == 0 and not result.stderr,
            "supported toolchain gate failed: " + result.stderr.strip())
    probe = parse_support_tsv(result.stdout, "support probe")
    keys(probe, {"os_id", "os_version_id", "architecture", "cc_path", "cc_version",
                 "llvm_version", "eshkol_repository", "eshkol_commit", "eshkol_version",
                 "eshkol_binary_sha256", "lock_sha256", "provenance_sha256"}, "support probe")
    return probe


def supported_snapshot(probe: dict[str, str]) -> dict[str, str]:
    lock_raw = (ROOT/"toolchain/eshkol.lock").read_bytes()
    provenance_path = Path(os.environ.get("ESHKOL_BUILD_DIR", ROOT/".deps/eshkol-build"))
    provenance_raw = (provenance_path/"eshkol-transformer-provenance.tsv").read_bytes()
    require(hashlib.sha256(lock_raw).hexdigest() == probe["lock_sha256"],
            "supported lock changed after gate")
    require(hashlib.sha256(provenance_raw).hexdigest() == probe["provenance_sha256"],
            "supported provenance changed after gate")
    try:
        lock_text, provenance_text = lock_raw.decode("utf-8"), provenance_raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise codec.BenchmarkFormatError("supported toolchain TSV is not UTF-8") from error
    lock = parse_support_tsv(lock_text, "lock")
    provenance = parse_support_tsv(provenance_text, "provenance")
    for name in ("eshkol_repository", "eshkol_commit", "eshkol_version", "llvm_version"):
        require(probe[name] == lock.get(name), f"support probe {name} mismatch")
    for name in ("cc_path", "cc_version", "eshkol_binary_sha256"):
        require(probe[name] == provenance.get(name), f"support probe {name} mismatch")
    require(probe["os_id"] == "ubuntu" and probe["os_version_id"] == "22.04" and
            probe["architecture"] == "x86_64", "unsupported probe host")
    return {"lock_tsv": lock_text, "provenance_tsv": provenance_text,
            "provenance_sha256": probe["provenance_sha256"]}


def run(definition_path: Path, report_path: Path, compiler: str, directory: Path,
        *, supported: bool = False) -> None:
    require(platform.system() == "Linux", "Linux pidfd/wait4 benchmark required")
    definition = load_definition(definition_path)
    probe = supported_probe(compiler) if supported else None
    support_identity = supported_snapshot(probe) if probe else None
    revision = common.git_output("rev-parse", "HEAD")
    if supported:
        require(not common.git_output("status", "--porcelain", "--untracked-files=all"),
                "supported project tree is dirty")
        compiler = probe["cc_path"]
    source_paths = SOURCES + (SUPPORT_SOURCES if supported else [])
    source_identity = {p: common.sha256_file(ROOT/p) for p in source_paths}
    target, helper, compiler_identity = build(compiler, directory)
    cases = []
    with tempfile.TemporaryDirectory(prefix="attention-v1.", dir=directory) as temporary:
        for row, shape in enumerate(SHAPES):
            for operation in definition["operations"]:
                samples, digests = [], []
                for repetition in range(definition["repetitions"]):
                    sample, digest = measure(helper, target, row, operation, definition, Path(temporary),
                                             f"{row}-{operation}-{repetition}")
                    samples.append({"repetition": repetition, "output_sha256": digest, **sample})
                    digests.append(digest)
                require(len(set(digests)) == 1, "nondeterministic output across measured repetitions")
                cases.append({"shape": shape, "operation": operation, "output_sha256": digests[0],
                              "samples": samples, "summary": summary(samples, definition["iterations"])})
    require(source_identity == {p: common.sha256_file(ROOT/p) for p in source_paths}, "sources changed during benchmark")
    if supported:
        require(supported_probe(compiler) == probe, "supported toolchain changed during benchmark")
        require(supported_snapshot(probe) == support_identity, "supported provenance changed during benchmark")
        require(common.sha256_file(Path(compiler)) == compiler_identity["sha256"],
                "supported compiler changed during benchmark")
        require(common.git_output("rev-parse", "HEAD") == revision and
                not common.git_output("status", "--porcelain", "--untracked-files=all"),
                "supported project tree changed during benchmark")
    report = {"stable": {"definition": definition, "definition_sha256": hashlib.sha256(encode_definition(definition)).hexdigest(),
                          "project": {"commit": revision,
                                      "dirty": bool(common.git_output("status", "--porcelain", "--untracked-files=all"))},
                          "sources": source_identity,
                          "compiler": compiler_identity,
                          "binaries": {"target": common.sha256_file(target), "helper": common.sha256_file(helper)}},
              "volatile": {"observed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z"),
                            "support_status": "supported" if supported else "compatibility-only",
                            "environment": {"architecture": platform.machine(), "kernel": platform.release(),
                                            "cpu_model": common.cpu_model(), "logical_cpus": os.cpu_count(),
                                            "os_release": common.os_release()}, "cases": cases}}
    if supported:
        report["stable"]["support_evidence"] = support_identity
        report["volatile"]["support_evidence"] = {
            name: probe[name] for name in ("os_id", "os_version_id", "architecture")}
    common.write_atomic(report_path, encode_report(report, version=2 if supported else 1))
    print(f"attention benchmark: PASS ({len(cases)} cases; {report['volatile']['support_status']}); report: {report_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    running = commands.add_parser("run")
    running.add_argument("definition", type=Path)
    running.add_argument("report", type=Path)
    running.add_argument("--cc", default="clang")
    running.add_argument("--supported", action="store_true")
    running.add_argument("--build-dir", type=Path, default=ROOT/"build/benchmarks/attention-v1")
    verifying = commands.add_parser("verify")
    verifying.add_argument("definition", type=Path)
    verifying.add_argument("report", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "run":
            run(args.definition.resolve(), args.report.resolve(), args.cc, args.build_dir.resolve(),
                supported=args.supported)
        else:
            definition, report = load_definition(args.definition), load_report(args.report)
            require(report["stable"]["definition"] == definition, "report does not match requested definition")
            print("attention benchmark report: PASS")
    except (codec.BenchmarkFormatError, common.BenchmarkRunError, OSError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
