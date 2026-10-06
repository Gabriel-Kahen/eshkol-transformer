"""Development-only exact T2 AOT workload runner; run inside pinned Docker.

Compile performance_runtime.esk separately against each source tree. This runner
checks every invocation against the independent Python reference, then records
whole-process time and peak RSS. It does not measure the merge kernel alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time

from tests.t2.reference import Core, Merge, encode_artifact, encode_bytes, pack_i64, train


SCENARIOS = (
    "encode-no-match", "encode-overlap", "encode-mixed-no-adjacency",
    "encode-sparse-chain", "encode-zero-merge",
    "encode-tiny", "train-many-docs", "train-one-doc",
    "train-near-limit-256",
)


def oracle() -> dict[str, str]:
    source = [bytes((document * 73 + index * 17) % 256 for index in range(16))
              for document in range(4096)]
    documents = tuple((part,) for part in source)
    trained = train(documents, 64, 2)
    joined = train(((b"".join(source),),), 64, 2)
    near_limit = train((((bytes(range(256)) * 256)[:-2],), (b"aa",)), 256, 1)
    sparse_model = Core(merges=tuple(
        Merge(rank, 256 + rank, 97 if rank == 0 else 255 + rank, 98)
        for rank in range(255)
    ))
    sparse_input = bytearray(b"c" * 65536)
    sparse_input[32768] = 97
    sparse_input[32769:33024] = b"b" * 255
    results = {
        "encode-no-match": (65536 * 8, hashlib.sha256(pack_i64((42,) * 65536)).hexdigest()),
        "encode-overlap": (32768 * 8, hashlib.sha256(pack_i64((256,) * 32768)).hexdigest()),
        "encode-mixed-no-adjacency":
            (65536 * 8, hashlib.sha256(pack_i64(tuple((index * 17) % 256
                                                     for index in range(65536)))).hexdigest()),
        "encode-sparse-chain":
            (65281 * 8,
             hashlib.sha256(pack_i64(encode_bytes(sparse_model, bytes(sparse_input)))).hexdigest()),
        "encode-zero-merge": (65536 * 8, hashlib.sha256(pack_i64((42,) * 65536)).hexdigest()),
        "encode-tiny": (2 * 8, hashlib.sha256(pack_i64((256, 97))).hexdigest()),
        "train-many-docs": (len(trained.merges), hashlib.sha256(encode_artifact(trained)).hexdigest()),
        "train-one-doc": (len(joined.merges), hashlib.sha256(encode_artifact(joined)).hexdigest()),
        "train-near-limit-256":
            (len(near_limit.merges), hashlib.sha256(encode_artifact(near_limit)).hexdigest()),
    }
    return {name: f"{name}\t{size}\t{digest}" for name, (size, digest) in results.items()}


def run(binary: Path, scenario: str, expected: str) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="t2-perf-") as directory:
        timing = Path(directory) / "time.tsv"
        started_ns = time.perf_counter_ns()
        result = subprocess.run(
            ["/usr/bin/time", "-f", "%e\t%M", "-o", str(timing),
             str(binary), scenario],
            capture_output=True, text=True, timeout=120, check=True,
        )
        elapsed_ns = time.perf_counter_ns() - started_ns
        actual = result.stdout.strip()
        if actual != expected:
            raise AssertionError(f"{scenario}: exact oracle mismatch: {actual!r} != {expected!r}")
        if result.stderr:
            raise AssertionError(f"{scenario}: unexpected runtime stderr: {result.stderr!r}")
        elapsed, rss = timing.read_text().strip().split("\t")
        return {"elapsed_ns": elapsed_ns, "gnu_time_seconds": float(elapsed),
                "peak_rss_kib": int(rss),
                "output": actual}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--core-source", required=True, type=Path)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--workload-source", required=True, type=Path)
    parser.add_argument("--compiler", required=True, type=Path)
    parser.add_argument("--compiler-sha256", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 3:
        parser.error("at least three repetitions are required")
    actual_core_hash = hashlib.sha256(args.core_source.read_bytes()).hexdigest()
    if actual_core_hash != args.source_sha256:
        parser.error("core source hash differs from declared source hash")
    actual_compiler_hash = hashlib.sha256(args.compiler.read_bytes()).hexdigest()
    if actual_compiler_hash != args.compiler_sha256:
        parser.error("compiler hash differs from declared compiler hash")
    expected = oracle()
    samples = {name: [] for name in SCENARIOS}
    for repetition in range(args.repetitions):
        for scenario in SCENARIOS:
            sample = run(args.binary, scenario, expected[scenario])
            samples[scenario].append(sample)
            print(f"{scenario} {repetition + 1}/{args.repetitions}: "
                  f"{sample['elapsed_ns'] / 1_000_000_000:.3f}s "
                  f"{sample['peak_rss_kib']} KiB", flush=True)
    report = {
        "schema": "t2-performance-development-v1",
        "source_commit": args.source_commit,
        "source_sha256": args.source_sha256,
        "core_source": str(args.core_source),
        "workload_source": str(args.workload_source),
        "workload_source_sha256": hashlib.sha256(args.workload_source.read_bytes()).hexdigest(),
        "compiler_sha256": args.compiler_sha256,
        "compiler": str(args.compiler),
        "image_digest": args.image_digest,
        "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        "repetitions": args.repetitions,
        "samples": samples,
        "measurement": "perf_counter_ns around GNU time and whole process; GNU time maximum resident KiB",
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
