"""Compare two verified, exact hosted P1 diagnostic receipts offline."""

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

from tests.ci.evidence import EvidenceError, strict_json


PAIR_BASE = "f9fd2de24ff10cd819a70b51f1697eb6e8fc3bfe"
PAIR_TRIGGER = "6cd70932351dadfb44f8471b1c047174e6c2ba7e"
SOURCE = "15f2522946b8ee13b9f166967a99f2665263d4d0"
TREE = "672e9da3d40061278e94ca0aa1954e98936afa7e"
JOB = "Supported parameter-state only (non-admission)"
SUITE = "Run complete parameter-state suite"
PASS = ("P1 PASS: E1B-integrated packaging, structural/native/registry checks, "
        "root-publication IR and poison proof, sanitizers, negatives, atomicity, and determinism")
MATCHED_FILES = (
    "host-os.txt", "source-inputs.sha256", "toolchain.txt", "toolchain-provenance.tsv",
)
TIMING = "validated-p1/compiler-timing.log"
TRACE_FILES = {f"validated-p1/pair-{pair}-memory.{suffix}"
               for pair in range(1, 7) for suffix in ("tsv", "summary.json")}
SHA = r"[0-9a-f]{40}"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def only_match(pattern, text, description):
    matches = re.findall(pattern, text, re.MULTILINE)
    require(len(matches) == 1, f"missing or ambiguous {description}")
    return matches[0]


def records(text):
    items = [line.split("\t") for line in text.splitlines()]
    require(all(len(item) == 2 and all(item) for item in items), "malformed TSV")
    require(len({item[0] for item in items}) == len(items), "duplicate TSV key")
    return dict(items)


def check_seal(root, paired):
    required = set(MATCHED_FILES) | {
        "identity.tsv", "collection.tsv", "initial-memory.txt", "parameter-state.log",
        "prerequisites.log", "toolchain.log", TIMING,
    }
    if paired:
        required |= TRACE_FILES
    require(root.is_dir() and not root.is_symlink(), "invalid receipt directory")
    files = {}
    for path in root.rglob("*"):
        require(not path.is_symlink(), f"linked receipt entry: {path}")
        name = path.relative_to(root).as_posix()
        if path.is_dir():
            require(name == "validated-p1", f"unexpected receipt directory: {name}")
        else:
            require(path.is_file(), f"invalid receipt entry: {name}")
            files[name] = path.read_bytes()
    seal = files.pop("SHA256SUMS").decode("ascii").splitlines()
    allowed = required | {"p1-step-summary.md"}
    require(required <= files.keys() <= allowed, "unexpected or missing receipt files")
    expected = [f"{hashlib.sha256(data).hexdigest()}  ./{name}"
                for name, data in sorted(files.items())]
    require(seal == expected, "receipt SHA256SUMS mismatch")
    return files


def check_receipt(root, paired, base, trigger):
    files = check_seal(root, paired)
    flag = int(paired)
    identity = records(files["identity.tsv"].decode("ascii"))
    expected = {
        "mode": f"P1_TWO_COMPILERS={flag} P1_PAIR_MEMORY_TRACE={flag} P1_KEEP_TMP=1",
        "status": "started", "workflow_base": base, "trigger_head": trigger,
        "source_head": SOURCE, "source_tree": TREE, "final_job_status": "success",
        **{f"{step}_outcome": "success" for step in
           ("toolchain", "prerequisites", "suite", "collect")},
    }
    require(all(identity.get(key) == value for key, value in expected.items()),
            "receipt identity or outcomes differ from the requested exact run")
    collection = records(files["collection.tsv"].decode("ascii"))
    require(collection == {
        "scope": ("P1 compiler process groups only" if paired else "serial P1 compiler timing only"),
        "suite_outcome": "success", "validated_pairs": "6" if paired else "0",
        "timing_outcome": "validated", "receipt_status": "complete",
    }, "receipt collection is incomplete")
    for key in ("run_id", "run_attempt"):
        require(re.fullmatch(r"[1-9][0-9]*", identity.get(key, "")), f"invalid {key}")
    return files, identity


def check_job(job_path, log_path, identity):
    job = strict_json(job_path.read_text(), "GitHub Actions job")
    require(isinstance(job, dict) and all(job.get(key) == value for key, value in {
        "name": JOB, "status": "completed", "conclusion": "success", "labels": ["ubuntu-22.04"],
        "runner_group_name": "GitHub Actions", "run_id": int(identity["run_id"]),
        "run_attempt": int(identity["run_attempt"]),
    }.items()), "job identity, runner class or conclusion does not match the receipt")
    require(type(job.get("id")) is int and job["id"] > 0, "invalid job ID")
    steps = [step for step in job["steps"] if step.get("name") == SUITE]
    require(len(steps) == 1 and steps[0].get("conclusion") == "success",
            "missing or failed complete suite step")
    start, end = [datetime.strptime(steps[0][key], "%Y-%m-%dT%H:%M:%SZ")
                  for key in ("started_at", "completed_at")]
    require(end > start, "invalid suite wall-time window")
    log = log_path.read_text(encoding="utf-8-sig")
    image = only_match(r"##\[group\]Runner Image\n[^\n]* Image: ubuntu-22\.04\n"
                       r"[^\n]* Version: ([0-9.]+)$", log, "Ubuntu runner image version")
    cache = only_match(r"(Cache restored from key|Cache not found for input keys): "
                       r"(eshkol-v1-Linux-X64-llvm-21\.1\.8-[0-9a-f]{64})$", log,
                       "toolchain cache restore state")
    return job["id"], int((end - start).total_seconds()), image, cache


def compiler_records(files):
    log = files["parameter-state.log"].decode("utf-8")
    require(log.splitlines().count(PASS) == 1, "missing or ambiguous final P1 PASS")
    scratch = only_match(
        r"^P1 preserved temporary evidence: (/home/runner/work/eshkol-transformer/"
        r"eshkol-transformer/\.deps/eshkol-transformer-p1\.[A-Za-z0-9]{6})$",
        log, "P1 scratch path")
    timing = files[TIMING].decode("ascii")
    lines = iter(timing.splitlines())
    commands = defaultdict(list)
    negatives = 0
    for line in lines:
        banner = line == "Command exited with non-zero status 1"
        row = re.fullmatch(r"elapsed_s=([0-9]+\.[0-9]{2}) max_rss_kib=([0-9]+) "
                           r"exit=([01]) command=(.+)", next(lines, "") if banner else line)
        require(row and (row[3] == "1") == banner, "invalid GNU time record")
        elapsed, rss, status, command = row.groups()
        scratch_cache = f"XDG_CACHE_HOME={scratch}/cache-"
        require(scratch_cache in command, "compiler cache path differs from the P1 scratch path")
        command = re.sub(re.escape(scratch_cache) + r"compiler\.[A-Za-z0-9]{6} ",
                         f"{scratch_cache}compiler.<random> ", command)
        commands[(status, command.replace(scratch, "<p1-scratch>"))].append([elapsed, int(rss)])
        negatives += banner
    require(sum(map(len, commands.values())) == 67 and negatives == 43, "compiler inventory changed")
    return commands


def compare(pair, serial, paired_job, serial_job, paired_log, serial_log, serial_base, serial_trigger):
    require(all(re.fullmatch(SHA, sha) for sha in (serial_base, serial_trigger)),
            "serial workflow base and trigger must be exact commit IDs")
    pair_files, pair_identity = check_receipt(pair, True, PAIR_BASE, PAIR_TRIGGER)
    serial_files, serial_identity = check_receipt(serial, False, serial_base, serial_trigger)
    pair_job_id, pair_seconds, pair_image, pair_cache = check_job(paired_job, paired_log, pair_identity)
    serial_job_id, serial_seconds, serial_image, serial_cache = check_job(
        serial_job, serial_log, serial_identity)
    require(pair_identity["run_id"] != serial_identity["run_id"] and pair_job_id != serial_job_id,
            "comparison requires distinct hosted runs and jobs")
    require(pair_image == serial_image, "runner image differs")
    require(pair_cache == serial_cache, "toolchain cache state differs")
    for name in MATCHED_FILES:
        require(pair_files[name] == serial_files[name], f"{name} differs")
    pair_commands = compiler_records(pair_files)
    serial_commands = compiler_records(serial_files)
    require({key: len(rows) for key, rows in pair_commands.items()}
            == {key: len(rows) for key, rows in serial_commands.items()}, "compiler commands differ")
    return {
        "source_head": SOURCE, "source_tree": TREE, "runner_image": pair_image, "cache_state": pair_cache,
        "scope": "Separate hosted jobs. Timing is observational. GNU time RSS is per command. "
                 "Paired RSS covers sampled compiler process groups only.",
        "paired": {"job_id": pair_job_id, "suite_wall_seconds": pair_seconds},
        "serial": {"job_id": serial_job_id, "suite_wall_seconds": serial_seconds},
        "initial_memory": [files["initial-memory.txt"].decode("ascii").strip()
                           for files in (pair_files, serial_files)],
        "compiler_records": [
            {"exit": int(status), "command": command, "paired": pair_commands[status, command],
             "serial": serial_commands[status, command]}
            for status, command in sorted(pair_commands)
        ],
        "compiler_measurement_columns": ["elapsed_seconds", "max_rss_kib"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pair", "serial"):
        parser.add_argument(f"--{name}", required=True, type=Path, help="Extracted receipt directory")
        parser.add_argument(f"--{name}-job", required=True, type=Path, help="Raw Actions job JSON")
        parser.add_argument(f"--{name}-log", required=True, type=Path, help="Raw Actions job log")
    parser.add_argument("--serial-base", required=True)
    parser.add_argument("--serial-trigger", required=True)
    args = parser.parse_args()
    try:
        result = compare(args.pair, args.serial, args.pair_job, args.serial_job, args.pair_log,
                         args.serial_log, args.serial_base, args.serial_trigger)
    except (EvidenceError, OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        parser.exit(1, f"P1 comparison unavailable: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
