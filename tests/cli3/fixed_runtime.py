"""Measure the retained PR #255 CLI3 CPU-f32 fixture on Ubuntu 22.04 x86-64."""

import argparse
from datetime import datetime, timezone
import filecmp
import json
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import struct
import subprocess
import sys
import tarfile
import tempfile

from tests.b0.run_benchmark import (
    BenchmarkRunError, cpu_model, parse_metrics, sha256_file, write_atomic,
)


ROOT = Path(__file__).resolve().parents[2]
PREPARE = ROOT / "scripts/prepare-cli3-b-fixed-training.sh"
VERIFY = ROOT / "scripts/run-cli3-downloaded-artifact.py"
MEASURE = ROOT / "benchmarks/measure_linux.c"
ARTIFACT = {
    "github_artifact_id": 15095929578,
    "github_artifact_sha256": "e8da9eba8e829a181e2e68af644737717ab28a1bd23b1a68ff309ec9348eed75",
    "github_run_id": 45773952998,
    "build_commit": "6ca6e432e565bcf1e5e0e3393ae969159f06c81d",
    "source_tree": "a00f10337f86e67e3ae29298d08da496d1d0f215",
    "archive_sha256": "d34de38737ef44d236c7c8147819fa828011eb014bf21e85f8037b08c3724144",
    "executable_sha256": "5eb2528b4b57a1659c8dc38bfb6bee10cb9a108583d85f38f09de3b61e657162",
}
FINAL_CHECKPOINT_SHA256 = "67ee14d747dfbaafeba1218eb03522b839c3d3339cd07290dc54eeefe12846ab"
ENV = {"PATH": os.defpath, "LC_ALL": "C", "TZ": "UTC",
       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
TRIALS = 5
TIMEOUT_MS = 30000


def unpack_binary(download: Path, source: Path, binary: Path) -> None:
    archive = download / "cli3-binary-artifact.tar"
    if sha256_file(archive) != ARTIFACT["archive_sha256"]:
        raise ValueError("archive differs from retained PR #255 artifact")
    subprocess.run([sys.executable, "-B", VERIFY, download, source, "--verify-only"],
                   check=True, env=ENV)
    with tarfile.open(archive, "r:") as tar, binary.open("xb") as dst:
        with tar.extractfile("cli3-binary-artifact/eshkol-transformer") as src:
            shutil.copyfileobj(src, dst)
    if sha256_file(binary) != ARTIFACT["executable_sha256"]:
        raise ValueError("extracted executable differs from retained PR #255 executable")
    binary.chmod(0o500)


def checkpoint_summary(stdout: Path, checkpoint: Path, delta: tuple, totals: tuple) -> dict:
    data = stdout.read_bytes()
    record = json.loads(data)
    numeric_keys = ("tokens", "updates", "epochs")
    bits_keys = ("loss_f32_bits", "mask_weight_f32_bits")
    if (data.count(b"\n") != 1 or not data.endswith(b"\n") or not isinstance(record, dict) or
            set(record) != {"checkpoint", "artifact", *numeric_keys, *bits_keys} or
            record.pop("checkpoint") != str(checkpoint) or
            record.get("artifact") != "checkpoint" or
            any(type(record[key]) is not int for key in numeric_keys) or
            tuple(record[key] for key in numeric_keys) != delta or
            any(not isinstance(record[key], str) or not re.fullmatch(r"[0-9a-f]{8}", record[key])
                for key in bits_keys)):
        raise ValueError(f"unexpected checkpoint summary in {stdout}")
    with checkpoint.open("rb") as stream:
        stream.seek(184)  # C2 v1 cumulative token, update and epoch counts.
        if struct.unpack("<QQQ", stream.read(24)) != totals:
            raise ValueError(f"unexpected cumulative counters in {checkpoint}")
    return record


def pretrain(binary: Path, fixture: Path, checkpoint: Path, delta: tuple, totals: tuple,
             *, resume: Path | None = None, helper: Path | None = None) -> tuple[dict, dict]:
    stdout = checkpoint.with_suffix(".json")
    stderr = checkpoint.with_suffix(".stderr")
    if any(os.path.lexists(path) for path in (checkpoint, stdout, stderr)):
        raise ValueError(f"pretrain output paths must be unused: {checkpoint}")
    command = [binary, "pretrain", "--config", fixture / "config.json",
               "--tokenizer", fixture / "byte.t1", "--train-corpus", fixture / "train",
               "--checkpoint", checkpoint]
    if resume is not None:
        command.extend(["--resume", resume])
    command.extend(["--max-updates", str(delta[1])])
    metrics = {}
    if helper is None:
        with stdout.open("xb") as out, stderr.open("xb") as err:
            subprocess.run(command, check=True, env=ENV, stdout=out, stderr=err,
                           timeout=TIMEOUT_MS / 1000)
    else:
        completed = subprocess.run([helper, str(TIMEOUT_MS), stdout, stderr, "--", *command],
                                   check=True, env=ENV, capture_output=True, text=True)
        if completed.stderr:
            raise ValueError(f"measurement helper emitted diagnostics: {completed.stderr}")
        metrics = parse_metrics(completed.stdout)
        if metrics["timed_out"] or metrics["exit_code"] or metrics["term_signal"]:
            raise ValueError(f"measurement failed for {checkpoint}: {metrics}")
    if stderr.stat().st_size:
        raise ValueError(f"pretrain emitted diagnostics: {stderr}")
    return checkpoint_summary(stdout, checkpoint, delta, totals), metrics


def measure(binary: Path, helper: Path, output: Path) -> dict:
    fixture = output / "fixture"
    fixture.mkdir()
    subprocess.run(["/usr/bin/bash", PREPARE, binary, fixture], check=True, env=ENV)
    chain = output / "chain"
    chain.mkdir()
    chain_cases = ((4, 1, 0), (3, 1, 1), (3, 1, 0), (4, 1, 1))
    chain_records = {}
    seed = chain / "update-1.c2"
    previous = None
    totals = (0, 0, 0)
    for index, delta in enumerate(chain_cases, 1):
        checkpoint = chain / f"update-{index}.c2"
        totals = tuple(a + b for a, b in zip(totals, delta))
        chain_records[str(index)], _ = pretrain(binary, fixture, checkpoint, delta, totals,
                                                resume=previous)
        previous = checkpoint
        if index == 1:
            seed_sha256 = sha256_file(seed)

    baseline = chain / "update-4.c2"
    final_sha256 = sha256_file(baseline)
    if final_sha256 != FINAL_CHECKPOINT_SHA256:
        raise ValueError("chained four-update checkpoint differs from the pinned digest")
    cases = {"pretrain-4": ((14, 4, 2), None), "resume-3": ((10, 3, 2), seed)}
    observations, records = [], {}
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    loadavg_before = os.getloadavg()
    for trial in range(1, TRIALS + 1):
        attempt = output / f"trial-{trial}"
        attempt.mkdir()
        order = list(cases) if trial % 2 else list(reversed(cases))
        for case in order:
            delta, prior = cases[case]
            checkpoint = attempt / f"{case}.c2"
            summary, metrics = pretrain(binary, fixture, checkpoint, delta, totals,
                                        resume=prior, helper=helper)
            if not filecmp.cmp(baseline, checkpoint, shallow=False):
                raise ValueError(f"four-update checkpoint bytes differ: {checkpoint}")
            if records.setdefault(case, summary) != summary:
                raise ValueError(f"invocation summary differs across trials: {checkpoint}")
            observations.append({"trial": trial, "case": case, **metrics})
    loadavg_after = os.getloadavg()
    if sha256_file(seed) != seed_sha256:
        raise ValueError("the resume seed changed during replay")
    ranges = {}
    for case in cases:
        ranges[case] = {}
        for field in ("elapsed_ns", "peak_rss_kib"):
            values = [entry[field] for entry in observations if entry["case"] == case]
            ranges[case][field] = {"min": min(values), "median": statistics.median(values),
                                   "max": max(values)}
    return {
        "deterministic": {"final_checkpoint_sha256": final_sha256,
                          "resume_seed_sha256": seed_sha256,
                          "final_checkpoint_totals": dict(zip(("tokens", "updates", "epochs"), totals)),
                          "path_independent_summaries": {"chain": chain_records, **records}},
        "observations": observations,
        "observed_ranges": ranges,
        "sample_window": {"started_at_utc": started_at, "loadavg_before": loadavg_before,
                          "loadavg_after": loadavg_after},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("download", type=Path)
    parser.add_argument("artifact_source", type=Path)
    parser.add_argument("output", type=Path, help="fresh directory for the report and output files")
    args = parser.parse_args()
    release = platform.freedesktop_os_release()
    if (sys.platform, platform.machine(), release.get("ID"), release.get("VERSION_ID")) != (
            "linux", "x86_64", "ubuntu", "22.04"):
        raise ValueError("CLI3 runtime measurement requires Ubuntu 22.04 x86-64")
    output = args.output.absolute()
    output.mkdir(mode=0o700)
    with tempfile.TemporaryDirectory(prefix=".tools-", dir=output) as temporary:
        binary = Path(temporary) / "eshkol-transformer"
        helper = Path(temporary) / "measure-linux"
        unpack_binary(args.download, args.artifact_source, binary)
        subprocess.run(["/usr/bin/cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                        MEASURE, "-o", helper], check=True, env=ENV)
        report = measure(binary, helper, output)
        report["measurement_helper_sha256"] = sha256_file(helper)
    report.update({
        "format": "eshkol-cli3-fixed-runtime-observation-v1",
        "artifact": ARTIFACT,
        "fixture_reference": "443dad4ba205f931211f821a86667c9cc91dd0e2:scripts/test-cli3-b.sh",
        "runner_inputs_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in
                                 (Path(__file__).resolve(), PREPARE, VERIFY, MEASURE,
                                  ROOT / "tests/b0/run_benchmark.py")},
        "environment": {"os_release": release["PRETTY_NAME"], "kernel": platform.release(),
                        "cpu": cpu_model(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
                        "storage_type": subprocess.check_output(
                            ["/usr/bin/stat", "-f", "-c", "%T", output], text=True).strip(),
                        "cli_environment": ENV},
        "measurement": {"scope": "end-to-end fixed CPU CLI3 child process",
                        "elapsed": "CLOCK_MONOTONIC before fork through pidfd completion",
                        "peak_rss": "Linux wait4 ru_maxrss for the direct child, KiB",
                        "timeout_ms": TIMEOUT_MS, "trials_per_case": TRIALS},
        "created_data": {
            str(path.relative_to(output)): {"size_bytes": path.stat().st_size,
                                            "sha256": sha256_file(path)}
            for path in sorted(output.rglob("*")) if path.is_file()},
    })
    report_file = output / "report.json"
    write_atomic(report_file, (json.dumps(report, indent=2, sort_keys=True) + "\n").encode())
    print(report_file)


if __name__ == "__main__":
    try:
        main()
    except (BenchmarkRunError, ValueError, OSError, struct.error, subprocess.SubprocessError) as exc:
        sys.exit(f"CLI3 fixed runtime measurement: {exc}")
