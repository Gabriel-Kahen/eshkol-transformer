#!/usr/bin/env python3
"""Run or validate the B0 native smoke benchmark."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__:
    from .benchmark_format import (
        BenchmarkFormatError,
        decimal_rate,
        definition_sha256,
        encode_report,
        load_definition,
        load_report,
        validate_report_against_definition,
    )
else:  # Direct script execution sets tests/b0 as the import root.
    from benchmark_format import (
        BenchmarkFormatError,
        decimal_rate,
        definition_sha256,
        encode_report,
        load_definition,
        load_report,
        validate_report_against_definition,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class BenchmarkRunError(RuntimeError):
    """The requested benchmark could not be measured exactly as specified."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tsv(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise BenchmarkRunError(f"cannot read required provenance {path}: {error}") from error
    for line_number, line in enumerate(lines, 1):
        fields = line.split("\t")
        if len(fields) != 2 or not fields[0] or not fields[1]:
            raise BenchmarkRunError(f"malformed TSV at {path}:{line_number}")
        key, value = fields
        if key in result:
            raise BenchmarkRunError(f"duplicate TSV key {key!r} in {path}")
        result[key] = value
    return result


def required(mapping: dict[str, str], key: str, path: Path) -> str:
    try:
        return mapping[key]
    except KeyError as error:
        raise BenchmarkRunError(f"missing {key!r} in {path}") from error


def git_output(*arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise BenchmarkRunError(f"cannot inspect project Git provenance: {error}") from error
    return result.stdout.strip()


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name") and ":" in line:
                value = line.split(":", 1)[1].strip()
                if value:
                    return value
    except (OSError, UnicodeError) as error:
        raise BenchmarkRunError(f"CPU identity is unavailable: {error}") from error
    raise BenchmarkRunError("CPU identity is unavailable in /proc/cpuinfo")


def os_release() -> str:
    try:
        values = load_tsv_like_os_release(Path("/etc/os-release"))
    except OSError as error:
        raise BenchmarkRunError(f"OS release metadata is unavailable: {error}") from error
    value = values.get("PRETTY_NAME")
    if not value:
        raise BenchmarkRunError("OS release metadata has no PRETTY_NAME")
    return value


def load_tsv_like_os_release(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1].replace(r"\"", '"').replace(r"\\", "\\")
        values[key] = value
    return values


def parse_metrics(raw: str) -> dict[str, Any]:
    fields: dict[str, str] = {}
    for line in raw.splitlines():
        parts = line.split("\t")
        if len(parts) != 2 or parts[0] in fields:
            raise BenchmarkRunError("measurement helper returned malformed metrics")
        fields[parts[0]] = parts[1]
    expected = {
        "format_version",
        "elapsed_ns",
        "peak_rss_kib",
        "exit_code",
        "term_signal",
        "timed_out",
    }
    if set(fields) != expected or fields["format_version"] != "1":
        raise BenchmarkRunError("measurement helper returned unsupported metric fields")
    try:
        metrics: dict[str, Any] = {
            "elapsed_ns": int(fields["elapsed_ns"]),
            "peak_rss_kib": int(fields["peak_rss_kib"]),
            "exit_code": int(fields["exit_code"]),
            "term_signal": int(fields["term_signal"]),
        }
    except ValueError as error:
        raise BenchmarkRunError("measurement helper returned a non-integer metric") from error
    if fields["timed_out"] not in {"true", "false"}:
        raise BenchmarkRunError("measurement helper returned an invalid timeout flag")
    metrics["timed_out"] = fields["timed_out"] == "true"
    if metrics["elapsed_ns"] <= 0 or metrics["peak_rss_kib"] <= 0:
        raise BenchmarkRunError("elapsed time and peak RSS must both be positive")
    return metrics


def measure_once(
    helper: Path,
    target: Path,
    arguments: list[str],
    timeout_ms: int,
    expected: dict[str, Any],
    temporary: Path,
    label: str,
) -> dict[str, Any]:
    stdout_path = temporary / f"{label}.stdout"
    stderr_path = temporary / f"{label}.stderr"
    try:
        completed = subprocess.run(
            [
                str(helper),
                str(timeout_ms),
                str(stdout_path),
                str(stderr_path),
                "--",
                str(target),
                *arguments,
            ],
            cwd=PROJECT_ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as error:
        raise BenchmarkRunError(f"measurement helper is unavailable: {error}") from error
    if completed.returncode != 0:
        diagnostic = completed.stderr.strip() or "no diagnostic"
        raise BenchmarkRunError(
            f"measurement helper failed with {completed.returncode}: {diagnostic}"
        )
    if completed.stderr:
        raise BenchmarkRunError(
            "measurement helper wrote unexpected stderr; refusing potentially "
            "degraded or fallback measurement"
        )
    metrics = parse_metrics(completed.stdout)
    if metrics["timed_out"]:
        raise BenchmarkRunError(f"benchmark {label} exceeded {timeout_ms} ms")
    if metrics["term_signal"] != 0:
        raise BenchmarkRunError(
            f"benchmark {label} terminated by signal {metrics['term_signal']}"
        )
    if metrics["exit_code"] != expected["exit_code"]:
        raise BenchmarkRunError(
            f"benchmark {label} exited {metrics['exit_code']}; expected {expected['exit_code']}"
        )
    if sha256_file(stdout_path) != expected["stdout_sha256"]:
        raise BenchmarkRunError(f"benchmark {label} stdout checksum mismatch")
    if expected["stderr"] == "empty" and stderr_path.stat().st_size != 0:
        raise BenchmarkRunError(f"benchmark {label} wrote unexpected stderr")
    metrics["stdout_sha256"] = sha256_file(stdout_path)
    metrics["stderr_sha256"] = sha256_file(stderr_path)
    return metrics


def build_report(
    definition_path: Path,
    helper: Path,
    provenance_path: Path,
    manifest_path: Path,
    support_status: str,
) -> dict[str, Any]:
    if platform.system() != "Linux":
        raise BenchmarkRunError("B0 measurement is unsupported outside Linux")
    definition = load_definition(definition_path)
    target_relative = Path(definition["argv"][0])
    target = (PROJECT_ROOT / target_relative).resolve()
    if not target.is_file() or not os.access(target, os.X_OK):
        raise BenchmarkRunError(f"benchmark target is missing or not executable: {target}")
    if not helper.is_file() or not os.access(helper, os.X_OK):
        raise BenchmarkRunError(f"measurement helper is missing or not executable: {helper}")

    provenance = load_tsv(provenance_path)
    manifest = load_tsv(manifest_path)
    if support_status not in {"supported", "compatibility-only"}:
        raise BenchmarkRunError("current support status is unavailable or unsupported")

    output_dir = manifest_path.parent / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    last_metrics: dict[str, Any] | None = None
    with tempfile.TemporaryDirectory(prefix="b0-smoke.", dir=output_dir) as temp_name:
        temporary = Path(temp_name)
        for index in range(definition["warmup"]):
            last_metrics = measure_once(
                helper,
                target,
                definition["argv"][1:],
                definition["timeout_ms"],
                definition["expected"],
                temporary,
                f"warmup-{index}",
            )
        for index in range(definition["repetitions"]):
            last_metrics = measure_once(
                helper,
                target,
                definition["argv"][1:],
                definition["timeout_ms"],
                definition["expected"],
                temporary,
                f"repetition-{index}",
            )
            elapsed_ns = last_metrics["elapsed_ns"]
            samples.append(
                {
                    "elapsed_ns": elapsed_ns,
                    "peak_rss_kib": last_metrics["peak_rss_kib"],
                    "repetition": index,
                    "throughput_per_second": decimal_rate(
                        definition["work"]["count"], elapsed_ns
                    ),
                    "work_count": definition["work"]["count"],
                }
            )
    if last_metrics is None:
        raise BenchmarkRunError("benchmark definition requested no executions")

    elapsed = sorted(sample["elapsed_ns"] for sample in samples)
    total_elapsed = sum(elapsed)
    total_work = definition["work"]["count"] * len(samples)
    logical_cpus = os.cpu_count()
    if logical_cpus is None or logical_cpus < 1:
        raise BenchmarkRunError("logical CPU count is unavailable")
    compiler = (
        f"{required(provenance, 'cc_path', provenance_path)} "
        f"{required(provenance, 'cc_version', provenance_path)}"
    )
    payload = {
        "stable": {
            "benchmark_id": definition["id"],
            "contract": {
                "backend": definition["backend"],
                "dtype": definition["dtype"],
                "repetitions": definition["repetitions"],
                "shapes": definition["shapes"],
                "warmup": definition["warmup"],
                "work": definition["work"],
            },
            "definition_sha256": definition_sha256(definition_path),
            "measurement_tool": {
                "binary_sha256": sha256_file(helper),
                "source_path": "benchmarks/measure_linux.c",
                "source_sha256": sha256_file(
                    PROJECT_ROOT / "benchmarks" / "measure_linux.c"
                ),
            },
            "outcome": {
                "exit_code": definition["expected"]["exit_code"],
                "stderr_sha256": last_metrics["stderr_sha256"],
                "stdout_sha256": last_metrics["stdout_sha256"],
            },
            "project": {
                "commit": git_output("rev-parse", "HEAD"),
                "dirty": bool(git_output("status", "--porcelain", "--untracked-files=all")),
            },
            "target": {"path": target_relative.as_posix(), "sha256": sha256_file(target)},
            "toolchain": {
                "binary_sha256": required(provenance, "eshkol_binary_sha256", provenance_path),
                "commit": required(provenance, "eshkol_commit", provenance_path),
                "repository": required(provenance, "eshkol_repository", provenance_path),
                "version": required(manifest, "eshkol_version", manifest_path),
            },
        },
        "volatile": {
            "environment": {
                "architecture": platform.machine(),
                "compiler": compiler,
                "cpu_model": cpu_model(),
                "kernel": platform.release(),
                "logical_cpus": logical_cpus,
                "os_release": os_release(),
            },
            "execution": {
                "samples": samples,
                "summary": {
                    "latency_ns_max": elapsed[-1],
                    "latency_ns_median": elapsed[(len(elapsed) - 1) // 2],
                    "latency_ns_min": elapsed[0],
                    "peak_rss_kib": max(sample["peak_rss_kib"] for sample in samples),
                    "throughput_per_second": decimal_rate(total_work, total_elapsed),
                },
            },
            "observed_at_utc": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "support_status": support_status,
        },
    }
    validate_report_against_definition(
        payload, definition, definition_sha256(definition_path)
    )
    return payload


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", dir=path.parent, delete=False
        ) as output:
            temporary = Path(output.name)
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except OSError as error:
        raise BenchmarkRunError(f"cannot write report atomically to {path}: {error}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def command_run(arguments: argparse.Namespace) -> None:
    definition_path = Path(arguments.definition).resolve()
    report_path = Path(arguments.report).resolve()
    payload = build_report(
        definition_path,
        Path(arguments.helper).resolve(),
        Path(arguments.provenance).resolve(),
        Path(arguments.toolchain_manifest).resolve(),
        arguments.support_status,
    )
    write_atomic(report_path, encode_report(payload))
    loaded = load_report(report_path)
    definition = load_definition(definition_path)
    validate_report_against_definition(
        loaded, definition, definition_sha256(definition_path)
    )
    summary = loaded["volatile"]["execution"]["summary"]
    print(f"report: {report_path}")
    print(f"support_status: {loaded['volatile']['support_status']}")
    print(f"latency_ns_median: {summary['latency_ns_median']}")
    print(f"peak_rss_kib: {summary['peak_rss_kib']}")
    print(f"throughput_per_second: {summary['throughput_per_second']}")


def command_verify(arguments: argparse.Namespace) -> None:
    definition_path = Path(arguments.definition).resolve()
    definition = load_definition(definition_path)
    report = load_report(Path(arguments.report).resolve())
    validate_report_against_definition(
        report, definition, definition_sha256(definition_path)
    )
    print("benchmark report: PASS")


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    subcommands = command.add_subparsers(dest="command", required=True)
    run = subcommands.add_parser("run")
    run.add_argument("definition")
    run.add_argument("report")
    run.add_argument("--helper", required=True)
    run.add_argument("--provenance", required=True)
    run.add_argument("--toolchain-manifest", required=True)
    run.add_argument(
        "--support-status",
        required=True,
        choices=("supported", "compatibility-only"),
    )
    run.set_defaults(function=command_run)
    verify = subcommands.add_parser("verify")
    verify.add_argument("definition")
    verify.add_argument("report")
    verify.set_defaults(function=command_verify)
    return command


def main() -> int:
    arguments = parser().parse_args()
    try:
        arguments.function(arguments)
    except (BenchmarkFormatError, BenchmarkRunError) as error:
        print(f"error: {error}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
