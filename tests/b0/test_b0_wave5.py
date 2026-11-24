"""Validator and test suite for the B0 Wave 5 extended benchmark report (wave5-bench-v1)."""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, NoReturn

SCHEMA_NAME = "wave5-bench-v1"
FORMAT_VERSION = 1
MAX_DOCUMENT_BYTES = 16 * 1024 * 1024
MAX_JSON_DEPTH = 32

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_UTC_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")

REQUIRED_GROUPS: dict[str, tuple[str, str, str]] = {
    "data4_pipeline": (
        "DATA4 Pipeline",
        "libeshkol_transformer_data4_pipeline.a",
        "data4_pipeline/libeshkol_transformer_data4_pipeline.a",
    ),
    "dpo": (
        "DPO/cDPO",
        "libeshkol_transformer_dpo.a",
        "dpo/libeshkol_transformer_dpo.a",
    ),
    "io5_quant": (
        "IO5 Quant Codec",
        "libeshkol_transformer_io5_quant.a",
        "io5_quant/libeshkol_transformer_io5_quant.a",
    ),
    "lora": (
        "LoRA",
        "libeshkol_transformer_lora.a",
        "lora/libeshkol_transformer_lora.a",
    ),
    "mod5_adamw": (
        "MOD5 AdamW",
        "libeshkol_transformer_mod5_adamw.a",
        "mod5_adamw/libeshkol_transformer_mod5_adamw.a",
    ),
    "mod5_mla": (
        "MOD5 MLA",
        "libeshkol_transformer_mod5_mla.a",
        "mod5_mla/libeshkol_transformer_mod5_mla.a",
    ),
    "mod5_moe": (
        "MOD5 MoE",
        "libeshkol_transformer_mod5_moe.a",
        "mod5_moe/libeshkol_transformer_mod5_moe.a",
    ),
    "mod5_rope": (
        "MOD5 RoPE",
        "libeshkol_transformer_mod5_rope.a",
        "mod5_rope/libeshkol_transformer_mod5_rope.a",
    ),
    "o2_muon": (
        "O2 Muon+WSD",
        "libeshkol_transformer_o2_muon.a",
        "o2_muon/libeshkol_transformer_o2_muon.a",
    ),
    "spec_decode": (
        "Speculative Decoding",
        "libeshkol_transformer_spec_decode.a",
        "spec_decode/libeshkol_transformer_spec_decode.a",
    ),
    "t2_trainer": (
        "T2 BPE Trainer",
        "libeshkol_transformer_t2_trainer.a",
        "t2_trainer/libeshkol_transformer_t2_trainer.a",
    ),
}

VALID_THROUGHPUT_UNITS = frozenset({
    "tokens_per_sec",
    "elems_per_sec",
    "docs_per_sec",
    "steps_per_sec",
})


class Wave5BenchmarkError(ValueError):
    """Malformed, non-canonical, tampered, or semantically invalid Wave 5 report."""


def _fail(message: str) -> NoReturn:
    raise Wave5BenchmarkError(message)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            _fail(f"duplicate JSON object key: {k!r}")
        out[k] = v
    return out


def _forbid_constant(token: str) -> NoReturn:
    _fail(f"non-finite or non-standard JSON constant is forbidden: {token}")


def _check_json_tree(value: Any, where: str = "$", depth: int = 0) -> None:
    if depth > MAX_JSON_DEPTH:
        _fail(f"{where}: JSON nesting exceeds {MAX_JSON_DEPTH}")
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        if value < -(2**63) or value > 2**63 - 1:
            _fail(f"{where}: integer outside signed 64-bit range")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(f"{where}: non-finite floating-point value is forbidden")
        return
    if isinstance(value, str):
        value.encode("utf-8", errors="strict")
        return
    if isinstance(value, list):
        for i, item in enumerate(value):
            _check_json_tree(item, f"{where}[{i}]", depth + 1)
        return
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                _fail(f"{where}: object key must be string")
            _check_json_tree(v, f"{where}.{k}", depth + 1)
        return
    _fail(f"{where}: unsupported JSON type {type(value).__name__}")


def canonical_json(value: Any) -> bytes:
    """Return canonical UTF-8 JSON bytes (sorted keys, compact separators, finite numbers)."""
    _check_json_tree(value)
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8", errors="strict")


def _obj(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{where}: expected object")
    return value


def _arr(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{where}: expected array")
    return value


def _keys(obj: Mapping[str, Any], expected: set[str], where: str) -> None:
    actual = set(obj.keys())
    if actual != expected:
        _fail(
            f"{where}: keys mismatch; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _str(value: Any, where: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        _fail(f"{where}: expected string")
    if nonempty and not value:
        _fail(f"{where}: expected non-empty string")
    return value


def _bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{where}: expected boolean")
    return value


def _int(value: Any, where: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{where}: expected integer")
    if value < minimum:
        _fail(f"{where}: integer {value} < {minimum}")
    return value


def _num(value: Any, where: str, *, strictly_positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{where}: expected finite number")
    val = float(value)
    if not math.isfinite(val):
        _fail(f"{where}: expected finite number")
    if strictly_positive and val <= 0.0:
        _fail(f"{where}: expected strictly positive number, got {val}")
    return val


def _sha256(value: Any, where: str) -> str:
    text = _str(value, where)
    if _SHA256_RE.fullmatch(text) is None:
        _fail(f"{where}: invalid lowercase 64-hex SHA-256")
    return text


def validate_document(doc: dict[str, Any], *, require_all_groups: bool = True) -> dict[str, Any]:
    _keys(
        doc,
        {
            "artifact_sha256",
            "environment",
            "groups",
            "measurement",
            "process",
            "schema",
            "toolchain",
            "version",
        },
        "$",
    )

    if doc["schema"] != SCHEMA_NAME:
        _fail(f"$.schema: unsupported schema {doc['schema']!r}")
    if _int(doc["version"], "$.version", 1) != FORMAT_VERSION:
        _fail(f"$.version: unsupported version {doc['version']!r}")
    _sha256(doc["artifact_sha256"], "$.artifact_sha256")

    # Environment
    env = _obj(doc["environment"], "$.environment")
    _keys(
        env,
        {"architecture", "cpu_model", "kernel", "logical_cpus", "observed_at_utc", "os_release"},
        "$.environment",
    )
    for k in ("architecture", "cpu_model", "kernel", "os_release"):
        _str(env[k], f"$.environment.{k}")
    _int(env["logical_cpus"], "$.environment.logical_cpus", 1)
    utc = _str(env["observed_at_utc"], "$.environment.observed_at_utc")
    if _UTC_RE.fullmatch(utc) is None:
        _fail("$.environment.observed_at_utc: expected YYYY-MM-DDTHH:MM:SSZ")
    datetime.fromisoformat(utc[:-1] + "+00:00")

    # Measurement
    meas = _obj(doc["measurement"], "$.measurement")
    _keys(
        meas,
        {
            "backend",
            "clock",
            "elapsed_unit",
            "limitations",
            "memory_source",
            "peak_rss_unit",
            "selected_group",
            "smoke_mode",
        },
        "$.measurement",
    )
    backend = _obj(meas["backend"], "$.measurement.backend")
    _keys(backend, {"acceleration", "device", "evidence", "name"}, "$.measurement.backend")
    if _bool(backend["acceleration"], "$.measurement.backend.acceleration"):
        _fail("$.measurement.backend.acceleration: CPU benchmark cannot claim acceleration")
    if backend["device"] != "cpu" or backend["name"] != "host-cpu":
        _fail("$.measurement.backend: must report host-cpu/cpu")
    if backend["evidence"] != "direct-native-c11-link":
        _fail("$.measurement.backend.evidence: unexpected evidence tag")
    if meas["clock"] != "clock_monotonic_raw":
        _fail("$.measurement.clock: expected 'clock_monotonic_raw'")
    if meas["elapsed_unit"] != "nanosecond":
        _fail("$.measurement.elapsed_unit: expected 'nanosecond'")
    if meas["memory_source"] != "linux_getrusage_rusage_self":
        _fail("$.measurement.memory_source: expected 'linux_getrusage_rusage_self'")
    if meas["peak_rss_unit"] != "kibibyte":
        _fail("$.measurement.peak_rss_unit: expected 'kibibyte'")
    selected_group = _str(meas["selected_group"], "$.measurement.selected_group")
    if selected_group != "all" and selected_group not in REQUIRED_GROUPS:
        _fail(f"$.measurement.selected_group: unknown group {selected_group!r}")
    _bool(meas["smoke_mode"], "$.measurement.smoke_mode")
    lims = _arr(meas["limitations"], "$.measurement.limitations")
    if not lims:
        _fail("$.measurement.limitations: must explicitly record limitations")
    for i, item in enumerate(lims):
        _str(item, f"$.measurement.limitations[{i}]")

    # Process
    proc = _obj(doc["process"], "$.process")
    _keys(proc, {"self_peak_rss_kib"}, "$.process")
    _int(proc["self_peak_rss_kib"], "$.process.self_peak_rss_kib", 1)

    # Toolchain & all 11 archive SHA-256s + K1
    tc = _obj(doc["toolchain"], "$.toolchain")
    _keys(
        tc,
        {"archives", "build_mode", "cflags", "compiler", "git_commit", "git_dirty", "k1_archive"},
        "$.toolchain",
    )
    if tc["build_mode"] not in {"normal", "sanitize"}:
        _fail("$.toolchain.build_mode: expected 'normal' or 'sanitize'")
    for k in ("cflags", "compiler", "git_commit"):
        _str(tc[k], f"$.toolchain.{k}")
    _bool(tc["git_dirty"], "$.toolchain.git_dirty")

    archives = _obj(tc["archives"], "$.toolchain.archives")
    _keys(archives, set(REQUIRED_GROUPS.keys()), "$.toolchain.archives")
    for gkey, (_, expected_filename, _) in REQUIRED_GROUPS.items():
        entry = _obj(archives[gkey], f"$.toolchain.archives.{gkey}")
        _keys(entry, {"filename", "sha256"}, f"$.toolchain.archives.{gkey}")
        if entry["filename"] != expected_filename:
            _fail(
                f"$.toolchain.archives.{gkey}.filename: expected {expected_filename!r}, "
                f"got {entry['filename']!r}"
            )
        _sha256(entry["sha256"], f"$.toolchain.archives.{gkey}.sha256")

    k1_entry = _obj(tc["k1_archive"], "$.toolchain.k1_archive")
    _keys(k1_entry, {"filename", "sha256"}, "$.toolchain.k1_archive")
    if k1_entry["filename"] != "libeshkol_transformer_k1.a":
        _fail("$.toolchain.k1_archive.filename: unexpected filename")
    _sha256(k1_entry["sha256"], "$.toolchain.k1_archive.sha256")

    # Groups & Cases
    groups = _obj(doc["groups"], "$.groups")
    if selected_group == "all":
        expected_groups = set(REQUIRED_GROUPS.keys())
    else:
        if require_all_groups:
            _fail(f"$.measurement.selected_group: expected 'all', got {selected_group!r}")
        expected_groups = {selected_group}
    _keys(groups, expected_groups, "$.groups")

    for gkey in sorted(expected_groups):
        expected_engine, expected_archive, _ = REQUIRED_GROUPS[gkey]
        gwhere = f"$.groups.{gkey}"
        gobj = _obj(groups[gkey], gwhere)
        _keys(gobj, {"archive", "cases", "engine"}, gwhere)
        if gobj["engine"] != expected_engine:
            _fail(f"{gwhere}.engine: expected {expected_engine!r}, got {gobj['engine']!r}")
        if gobj["archive"] != expected_archive:
            _fail(f"{gwhere}.archive: expected {expected_archive!r}, got {gobj['archive']!r}")
        cases = _arr(gobj["cases"], f"{gwhere}.cases")
        if not cases:
            _fail(f"{gwhere}.cases: expected non-empty cases")
        seen_names: set[str] = set()
        for i, raw_case in enumerate(cases):
            cwhere = f"{gwhere}.cases[{i}]"
            c = _obj(raw_case, cwhere)
            _keys(
                c,
                {
                    "checksum",
                    "iterations",
                    "max_ns",
                    "mean_ns",
                    "median_ns",
                    "median_ns_per_iter",
                    "min_ns",
                    "name",
                    "operation",
                    "output_sha256",
                    "p95_ns",
                    "samples",
                    "shape",
                    "throughput",
                    "throughput_unit",
                    "warmup",
                    "work_units_per_iter",
                },
                cwhere,
            )
            cname = _str(c["name"], f"{cwhere}.name")
            if cname in seen_names:
                _fail(f"{cwhere}.name: duplicate case name {cname!r}")
            seen_names.add(cname)
            _str(c["operation"], f"{cwhere}.operation")
            _str(c["shape"], f"{cwhere}.shape")
            unit = _str(c["throughput_unit"], f"{cwhere}.throughput_unit")
            if unit not in VALID_THROUGHPUT_UNITS:
                _fail(f"{cwhere}.throughput_unit: invalid unit {unit!r}")

            checksum = _int(c["checksum"], f"{cwhere}.checksum", 1)
            if checksum == 0:
                _fail(f"{cwhere}.checksum: must be non-zero")
            out_sha = _sha256(c["output_sha256"], f"{cwhere}.output_sha256")
            expected_checksum = (int(out_sha[:16], 16) & 0x3FFFFFFFFFFFFFFF) or 1
            if checksum != expected_checksum:
                _fail(
                    f"{cwhere}.checksum: mismatch with output_sha256 prefix "
                    f"({checksum} != {expected_checksum})"
                )

            _int(c["warmup"], f"{cwhere}.warmup", 1)
            _int(c["samples"], f"{cwhere}.samples", 1)
            iters = _int(c["iterations"], f"{cwhere}.iterations", 1)
            work = _int(c["work_units_per_iter"], f"{cwhere}.work_units_per_iter", 1)

            min_ns = _int(c["min_ns"], f"{cwhere}.min_ns", 1)
            med_ns = _int(c["median_ns"], f"{cwhere}.median_ns", 1)
            p95_ns = _int(c["p95_ns"], f"{cwhere}.p95_ns", 1)
            max_ns = _int(c["max_ns"], f"{cwhere}.max_ns", 1)
            if not (min_ns <= med_ns <= p95_ns <= max_ns):
                _fail(
                    f"{cwhere}: timing order violated "
                    f"({min_ns} <= {med_ns} <= {p95_ns} <= {max_ns})"
                )
            mean_ns = _num(c["mean_ns"], f"{cwhere}.mean_ns", strictly_positive=True)
            if mean_ns < min_ns - 1e-3 or mean_ns > max_ns + 1e-3:
                _fail(f"{cwhere}.mean_ns: {mean_ns} outside [{min_ns}, {max_ns}]")
            per_iter = _num(
                c["median_ns_per_iter"],
                f"{cwhere}.median_ns_per_iter",
                strictly_positive=True,
            )
            if abs(per_iter - (med_ns / iters)) > max(1e-2, 1e-5 * (med_ns / iters)):
                _fail(f"{cwhere}.median_ns_per_iter: inconsistent with median_ns/iterations")
            tput = _num(c["throughput"], f"{cwhere}.throughput", strictly_positive=True)
            expected_tput = (work * iters * 1e9) / med_ns
            if abs(tput - expected_tput) > max(1.0, 1e-4 * expected_tput):
                _fail(f"{cwhere}.throughput: {tput} inconsistent with expected {expected_tput}")

    return doc


def load_and_verify_report(
    source: bytes | bytearray | str | Path,
    *,
    require_all_groups: bool = True,
) -> dict[str, Any]:
    raw = Path(source).read_bytes() if isinstance(source, (str, Path)) else bytes(source)
    if len(raw) > MAX_DOCUMENT_BYTES:
        _fail(f"report exceeds {MAX_DOCUMENT_BYTES} bytes")
    if raw.startswith(b"\xef\xbb\xbf"):
        _fail("report must not contain a UTF-8 BOM")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise Wave5BenchmarkError("report is not valid UTF-8") from exc
    try:
        doc = json.loads(
            text,
            object_pairs_hook=_pairs,
            parse_constant=_forbid_constant,
        )
    except json.JSONDecodeError as exc:
        raise Wave5BenchmarkError(f"invalid JSON: {exc.msg}") from exc
    doc = _obj(doc, "$")
    if raw != canonical_json(doc) + b"\n":
        _fail("report bytes are not canonical UTF-8 JSON with single trailing LF")
    if "artifact_sha256" not in doc:
        _fail("missing required top-level artifact_sha256")
    expected_digest = _sha256(doc["artifact_sha256"], "$.artifact_sha256")
    body = {k: v for k, v in doc.items() if k != "artifact_sha256"}
    actual_digest = hashlib.sha256(canonical_json(body)).hexdigest()
    if not hmac.compare_digest(expected_digest, actual_digest):
        _fail("artifact_sha256 mismatch over canonical body")
    return validate_document(doc, require_all_groups=require_all_groups)


def encode_report(body: Mapping[str, Any]) -> bytes:
    clean_body = {k: v for k, v in body.items() if k != "artifact_sha256"}
    digest = hashlib.sha256(canonical_json(clean_body)).hexdigest()
    doc = dict(clean_body)
    doc["artifact_sha256"] = digest
    return canonical_json(doc) + b"\n"


def extract_checksums(doc: Mapping[str, Any]) -> dict[tuple[str, str], tuple[int, str]]:
    out: dict[tuple[str, str], tuple[int, str]] = {}
    for gkey, gobj in doc["groups"].items():
        for case in gobj["cases"]:
            out[(gkey, case["name"])] = (case["checksum"], case["output_sha256"])
    return out


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_report_path() -> Path:
    env_path = os.environ.get("WAVE5_BENCHMARK_REPORT")
    if env_path:
        return Path(env_path)
    return _repo_root() / "build" / "b0_wave5" / "wave5-bench-v1-report.json"


class Wave5BenchmarkTests(unittest.TestCase):
    def _load_raw(self) -> bytes:
        path = _default_report_path()
        self.assertTrue(path.is_file(), f"expected generated benchmark report at {path}")
        return path.read_bytes()

    def test_emitted_report_passes_full_validation(self) -> None:
        doc = load_and_verify_report(self._load_raw())
        self.assertEqual(doc["schema"], SCHEMA_NAME)
        self.assertEqual(set(doc["groups"].keys()), set(REQUIRED_GROUPS.keys()))
        self.assertEqual(len(doc["groups"]), 11)
        self.assertGreater(doc["process"]["self_peak_rss_kib"], 0)
        # Verify archive SHA-256s match disk when archive dir is present.
        archive_dir_env = os.environ.get("WAVE5_ARCHIVE_DIR")
        if archive_dir_env:
            archive_dir = Path(archive_dir_env)
            for gkey, (_, _, relpath) in REQUIRED_GROUPS.items():
                disk_sha = hashlib.sha256((archive_dir / relpath).read_bytes()).hexdigest()
                self.assertEqual(doc["toolchain"]["archives"][gkey]["sha256"], disk_sha)
            k1_sha = hashlib.sha256(
                (archive_dir / "k1/libeshkol_transformer_k1.a").read_bytes()
            ).hexdigest()
            self.assertEqual(doc["toolchain"]["k1_archive"]["sha256"], k1_sha)

    def test_two_run_checksum_invariance(self) -> None:
        run2_env = os.environ.get("WAVE5_BENCHMARK_REPORT_2")
        if not run2_env:
            self.skipTest("WAVE5_BENCHMARK_REPORT_2 not provided")
        doc1 = load_and_verify_report(self._load_raw())
        doc2 = load_and_verify_report(Path(run2_env))
        self.assertEqual(extract_checksums(doc1), extract_checksums(doc2))

    def test_sanitize_smoke_matches_normal_checksums(self) -> None:
        san_env = os.environ.get("WAVE5_BENCHMARK_REPORT_SAN")
        if not san_env:
            self.skipTest("WAVE5_BENCHMARK_REPORT_SAN not provided")
        doc_norm = load_and_verify_report(self._load_raw())
        doc_san = load_and_verify_report(Path(san_env))
        self.assertEqual(doc_san["toolchain"]["build_mode"], "sanitize")
        self.assertTrue(doc_san["measurement"]["smoke_mode"])
        self.assertEqual(extract_checksums(doc_norm), extract_checksums(doc_san))

    def test_single_group_report_validates(self) -> None:
        grp_env = os.environ.get("WAVE5_BENCHMARK_REPORT_GROUP")
        if not grp_env:
            self.skipTest("WAVE5_BENCHMARK_REPORT_GROUP not provided")
        doc = load_and_verify_report(Path(grp_env), require_all_groups=False)
        self.assertEqual(doc["measurement"]["selected_group"], "lora")
        self.assertEqual(set(doc["groups"].keys()), {"lora"})
        with self.assertRaisesRegex(Wave5BenchmarkError, "expected 'all'"):
            load_and_verify_report(Path(grp_env), require_all_groups=True)

    def test_tampered_single_byte_fails_checksum(self) -> None:
        raw = bytearray(self._load_raw())
        idx = raw.index(b'"min_ns":') + len(b'"min_ns":')
        # Flip one ASCII digit so JSON remains valid while checksum breaks.
        raw[idx] = ord("9") if raw[idx] != ord("9") else ord("8")
        with self.assertRaisesRegex(Wave5BenchmarkError, "artifact_sha256 mismatch"):
            load_and_verify_report(bytes(raw))

    def test_wrong_schema_rejected(self) -> None:
        doc = json.loads(self._load_raw().decode("utf-8"))
        doc["schema"] = "wave5-bench-v2"
        with self.assertRaisesRegex(Wave5BenchmarkError, "unsupported schema"):
            load_and_verify_report(encode_report(doc))

    def test_missing_group_rejected(self) -> None:
        doc = json.loads(self._load_raw().decode("utf-8"))
        del doc["groups"]["mod5_mla"]
        with self.assertRaisesRegex(Wave5BenchmarkError, "keys mismatch"):
            load_and_verify_report(encode_report(doc))

    def test_timing_order_violation_rejected(self) -> None:
        doc = json.loads(self._load_raw().decode("utf-8"))
        case = doc["groups"]["lora"]["cases"][0]
        case["p95_ns"] = case["max_ns"] + 1
        with self.assertRaisesRegex(Wave5BenchmarkError, "timing order violated"):
            load_and_verify_report(encode_report(doc))

    def test_zero_checksum_rejected(self) -> None:
        doc = json.loads(self._load_raw().decode("utf-8"))
        doc["groups"]["dpo"]["cases"][0]["checksum"] = 0
        with self.assertRaisesRegex(Wave5BenchmarkError, "integer 0 < 1"):
            load_and_verify_report(encode_report(doc))

    def test_non_finite_json_constant_rejected(self) -> None:
        raw = self._load_raw().decode("utf-8")
        for bad in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(bad=bad):
                mutated = re.sub(r'"mean_ns":[0-9.eE+-]+', f'"mean_ns":{bad}', raw, count=1)
                with self.assertRaisesRegex(Wave5BenchmarkError, "non-finite"):
                    load_and_verify_report(mutated.encode("utf-8"))

    def test_non_canonical_whitespace_rejected(self) -> None:
        doc = json.loads(self._load_raw().decode("utf-8"))
        pretty = json.dumps(doc, indent=2, sort_keys=True) + "\n"
        with self.assertRaisesRegex(Wave5BenchmarkError, "not canonical"):
            load_and_verify_report(pretty.encode("utf-8"))

    def test_invalid_cli_flags_fail_closed(self) -> None:
        bin_env = os.environ.get("WAVE5_BENCHMARK_BIN")
        bench_bin = Path(bin_env) if bin_env else _repo_root() / "build" / "b0_wave5" / "b0_wave5_bench"
        if not bench_bin.is_file():
            self.skipTest(f"benchmark binary not present at {bench_bin}")
        with tempfile.TemporaryDirectory() as tmp:
            out_path = str(Path(tmp) / "should_not_exist.json")
            bad_invocations = [
                [],  # missing --output
                ["--output", out_path, "--unknown-flag"],
                ["--output", out_path, "--smoke", "--full"],
                ["--output", out_path, "--group", "nonexistent_engine"],
                ["--output", out_path, "--build-mode", "turbo"],
                ["--output", out_path, "--git-dirty", "2"],
                ["--output", out_path, "--archive-dir", str(Path(tmp) / "missing_archives")],
            ]
            for args in bad_invocations:
                with self.subTest(args=args):
                    proc = subprocess.run(
                        [str(bench_bin), *args],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertNotEqual(proc.returncode, 0, f"expected non-zero exit for {args}")
                    self.assertFalse(Path(out_path).exists(), "failed CLI must not create output")


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "--verify":
        doc = load_and_verify_report(argv[2], require_all_groups=False)
        total_cases = sum(len(g["cases"]) for g in doc["groups"].values())
        print(
            f"wave5-bench-v1 VERIFY PASS: mode={doc['toolchain']['build_mode']} "
            f"smoke={doc['measurement']['smoke_mode']} "
            f"group={doc['measurement']['selected_group']} "
            f"groups={len(doc['groups'])} cases={total_cases} "
            f"rss_kib={doc['process']['self_peak_rss_kib']} "
            f"sha256={doc['artifact_sha256']}"
        )
        return 0
    return 1


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--verify":
        raise SystemExit(main(sys.argv))
    unittest.main()
