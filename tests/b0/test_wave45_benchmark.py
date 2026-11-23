"""Validator and test suite for the B0 Wave 4/5 unified benchmark report (wave45-bench-v1)."""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
import os
import re
import sys
import unittest
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, NoReturn

FORMAT_NAME = "wave45-bench-v1"
FORMAT_VERSION = 1
MAX_DOCUMENT_BYTES = 16 * 1024 * 1024
MAX_JSON_DEPTH = 32

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_UTC_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")

REQUIRED_LIBRARIES = [
    "libeshkol_transformer_k1.a",
    "libeshkol_transformer_mod5.a",
    "libeshkol_transformer_pa4_wider.a",
    "libeshkol_transformer_mp4.a",
    "libeshkol_ac4.a",
    "libeshkol_dist5.a",
]


class Wave45BenchmarkError(ValueError):
    """Malformed, non-canonical, tampered, or semantically invalid Wave 4/5 report."""


def _fail(message: str) -> NoReturn:
    raise Wave45BenchmarkError(message)


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


def _num(value: Any, where: str, *, strictly_positive: bool = False, non_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{where}: expected finite number")
    val = float(value)
    if not math.isfinite(val):
        _fail(f"{where}: expected finite number")
    if strictly_positive and val <= 0.0:
        _fail(f"{where}: expected strictly positive number, got {val}")
    if non_negative and val < 0.0:
        _fail(f"{where}: expected non-negative number, got {val}")
    return val


def _sha256(value: Any, where: str) -> str:
    text = _str(value, where)
    if _SHA256_RE.fullmatch(text) is None:
        _fail(f"{where}: invalid lowercase 64-hex SHA-256")
    return text


def _shape(value: Any, where: str, expected_len: int | None = None) -> list[int]:
    arr = _arr(value, where)
    if not arr:
        _fail(f"{where}: shape must be non-empty")
    if expected_len is not None and len(arr) != expected_len:
        _fail(f"{where}: expected shape rank {expected_len}, got {len(arr)}")
    return [_int(v, f"{where}[{i}]", 1) for i, v in enumerate(arr)]


def _validate_timing_triplet(
    obj: Mapping[str, Any],
    where: str,
    prefix: str = "",
) -> tuple[int, int, int, float]:
    min_ns = _int(obj[f"{prefix}min_ns"], f"{where}.{prefix}min_ns", 1)
    med_ns = _int(obj[f"{prefix}median_ns"], f"{where}.{prefix}median_ns", 1)
    max_ns = _int(obj[f"{prefix}max_ns"], f"{where}.{prefix}max_ns", 1)
    if not (min_ns <= med_ns <= max_ns):
        _fail(f"{where}: latency ordering violated ({min_ns} <= {med_ns} <= {max_ns})")
    per_iter = _num(
        obj[f"{prefix}median_ns_per_iter"],
        f"{where}.{prefix}median_ns_per_iter",
        strictly_positive=True,
    )
    return min_ns, med_ns, max_ns, per_iter


def validate_payload(raw: Any) -> dict[str, Any]:
    payload = _obj(raw, "$.payload")
    _keys(
        payload,
        {
            "ac4",
            "dist5",
            "environment",
            "measurement",
            "mod5",
            "mp4",
            "pa4_wider",
            "process",
            "toolchain",
        },
        "$.payload",
    )

    # Environment
    env = _obj(payload["environment"], "$.payload.environment")
    _keys(
        env,
        {"architecture", "cpu_model", "kernel", "logical_cpus", "observed_at_utc", "os_release"},
        "$.payload.environment",
    )
    for k in ("architecture", "cpu_model", "kernel", "os_release"):
        _str(env[k], f"$.payload.environment.{k}")
    _int(env["logical_cpus"], "$.payload.environment.logical_cpus", 1)
    utc = _str(env["observed_at_utc"], "$.payload.environment.observed_at_utc")
    if _UTC_RE.fullmatch(utc) is None:
        _fail("$.payload.environment.observed_at_utc: expected YYYY-MM-DDTHH:MM:SSZ")
    datetime.fromisoformat(utc[:-1] + "+00:00")

    # Measurement
    meas = _obj(payload["measurement"], "$.payload.measurement")
    _keys(
        meas,
        {
            "backend",
            "clock",
            "elapsed_unit",
            "limitations",
            "memory_source",
            "peak_rss_unit",
            "smoke_mode",
        },
        "$.payload.measurement",
    )
    backend = _obj(meas["backend"], "$.payload.measurement.backend")
    _keys(backend, {"acceleration", "device", "evidence", "name"}, "$.payload.measurement.backend")
    if _bool(backend["acceleration"], "$.payload.measurement.backend.acceleration"):
        _fail("$.payload.measurement.backend.acceleration: CPU benchmark cannot claim acceleration")
    if backend["device"] != "cpu" or backend["name"] != "host-cpu":
        _fail("$.payload.measurement.backend: must report host-cpu/cpu")
    if backend["evidence"] != "direct-native-c11-link":
        _fail("$.payload.measurement.backend.evidence: unexpected evidence tag")
    if meas["clock"] != "clock_monotonic":
        _fail("$.payload.measurement.clock: expected 'clock_monotonic'")
    if meas["elapsed_unit"] != "nanosecond":
        _fail("$.payload.measurement.elapsed_unit: expected 'nanosecond'")
    if meas["memory_source"] != "linux_getrusage_rusage_self":
        _fail("$.payload.measurement.memory_source: expected 'linux_getrusage_rusage_self'")
    if meas["peak_rss_unit"] != "kibibyte":
        _fail("$.payload.measurement.peak_rss_unit: expected 'kibibyte'")
    _bool(meas["smoke_mode"], "$.payload.measurement.smoke_mode")
    lims = _arr(meas["limitations"], "$.payload.measurement.limitations")
    if not lims:
        _fail("$.payload.measurement.limitations: must explicitly record limitations")
    for i, item in enumerate(lims):
        _str(item, f"$.payload.measurement.limitations[{i}]")

    # Process RSS
    proc = _obj(payload["process"], "$.payload.process")
    _keys(proc, {"children_peak_rss_kib", "self_peak_rss_kib"}, "$.payload.process")
    _int(proc["self_peak_rss_kib"], "$.payload.process.self_peak_rss_kib", 1)
    _int(proc["children_peak_rss_kib"], "$.payload.process.children_peak_rss_kib", 0)

    # Toolchain
    tc = _obj(payload["toolchain"], "$.payload.toolchain")
    _keys(tc, {"build_mode", "cflags", "compiler", "git_commit", "git_dirty", "libraries"}, "$.payload.toolchain")
    if tc["build_mode"] not in {"normal", "sanitize"}:
        _fail("$.payload.toolchain.build_mode: expected 'normal' or 'sanitize'")
    for k in ("cflags", "compiler", "git_commit"):
        _str(tc[k], f"$.payload.toolchain.{k}")
    _bool(tc["git_dirty"], "$.payload.toolchain.git_dirty")
    libs = _arr(tc["libraries"], "$.payload.toolchain.libraries")
    if libs != REQUIRED_LIBRARIES:
        _fail(f"$.payload.toolchain.libraries: expected {REQUIRED_LIBRARIES}, got {libs}")

    # 1. MOD5
    mod5 = _obj(payload["mod5"], "$.payload.mod5")
    _keys(mod5, {"cases", "provider"}, "$.payload.mod5")
    if mod5["provider"] != "mod5.cpu-f32.serial":
        _fail("$.payload.mod5.provider: unexpected provider")
    mod5_cases = _arr(mod5["cases"], "$.payload.mod5.cases")
    if not mod5_cases:
        _fail("$.payload.mod5.cases: expected non-empty cases")
    seen_mod5_kernels: set[str] = set()
    for i, raw_case in enumerate(mod5_cases):
        where = f"$.payload.mod5.cases[{i}]"
        c = _obj(raw_case, where)
        _keys(
            c,
            {
                "iterations",
                "kernel",
                "max_ns",
                "median_ns",
                "median_ns_per_iter",
                "min_ns",
                "operation",
                "output_sha256",
                "samples",
                "shape",
                "warmup",
            },
            where,
        )
        kernel = _str(c["kernel"], f"{where}.kernel")
        if kernel not in {"rmsnorm", "swiglu", "variant_attention_mqa", "variant_attention_gqa"}:
            _fail(f"{where}.kernel: unexpected kernel {kernel!r}")
        seen_mod5_kernels.add(kernel)
        if c["operation"] not in {"forward", "backward", "forward_backward"}:
            _fail(f"{where}.operation: invalid operation")
        _int(c["iterations"], f"{where}.iterations", 1)
        _int(c["samples"], f"{where}.samples", 1)
        _int(c["warmup"], f"{where}.warmup", 1)
        _shape(c["shape"], f"{where}.shape", 3 if kernel in {"rmsnorm", "swiglu"} else 6)
        _validate_timing_triplet(c, where)
        _sha256(c["output_sha256"], f"{where}.output_sha256")
    if seen_mod5_kernels != {"rmsnorm", "swiglu", "variant_attention_mqa", "variant_attention_gqa"}:
        _fail(f"$.payload.mod5.cases: missing required kernels: {seen_mod5_kernels}")

    # 2. PA4-Wider
    pa4 = _obj(payload["pa4_wider"], "$.payload.pa4_wider")
    _keys(pa4, {"cases", "providers"}, "$.payload.pa4_wider")
    provs = _obj(pa4["providers"], "$.payload.pa4_wider.providers")
    _keys(provs, {"online", "ref"}, "$.payload.pa4_wider.providers")
    if provs["ref"] != "pa4-wider.ref" or provs["online"] != "pa4-wider.online":
        _fail("$.payload.pa4_wider.providers: unexpected provider names")
    pa4_cases = _arr(pa4["cases"], "$.payload.pa4_wider.cases")
    seen_pa4_t_op: set[tuple[int, str]] = set()
    for i, raw_case in enumerate(pa4_cases):
        where = f"$.payload.pa4_wider.cases[{i}]"
        c = _obj(raw_case, where)
        _keys(
            c,
            {
                "ab_ratio_max",
                "ab_ratio_min",
                "ba_ratio_max",
                "ba_ratio_min",
                "iterations",
                "max_abs_diff",
                "max_rel_diff",
                "online_max_ns",
                "online_median_ns",
                "online_median_ns_per_iter",
                "online_min_ns",
                "online_output_sha256",
                "online_vs_ref_ratio",
                "operation",
                "pairs",
                "parity_verified",
                "ref_max_ns",
                "ref_median_ns",
                "ref_median_ns_per_iter",
                "ref_min_ns",
                "ref_output_sha256",
                "scope",
                "shape",
                "speedup_pct",
                "warmup",
            },
            where,
        )
        if c["operation"] not in {"forward", "backward", "joint"}:
            _fail(f"{where}.operation: invalid operation")
        if c["scope"] != "dispatch":
            _fail(f"{where}.scope: expected 'dispatch'")
        if not _bool(c["parity_verified"], f"{where}.parity_verified"):
            _fail(f"{where}.parity_verified: must be true")
        shape = _shape(c["shape"], f"{where}.shape", 6)
        seen_pa4_t_op.add((shape[3], c["operation"]))
        _int(c["pairs"], f"{where}.pairs", 2)
        _int(c["iterations"], f"{where}.iterations", 1)
        _int(c["warmup"], f"{where}.warmup", 1)
        _validate_timing_triplet(c, where, "ref_")
        _validate_timing_triplet(c, where, "online_")
        _num(c["online_vs_ref_ratio"], f"{where}.online_vs_ref_ratio", strictly_positive=True)
        _num(c["speedup_pct"], f"{where}.speedup_pct")
        ab_min = _num(c["ab_ratio_min"], f"{where}.ab_ratio_min", strictly_positive=True)
        ab_max = _num(c["ab_ratio_max"], f"{where}.ab_ratio_max", strictly_positive=True)
        ba_min = _num(c["ba_ratio_min"], f"{where}.ba_ratio_min", strictly_positive=True)
        ba_max = _num(c["ba_ratio_max"], f"{where}.ba_ratio_max", strictly_positive=True)
        if ab_min > ab_max or ba_min > ba_max:
            _fail(f"{where}: AB/BA ratio min exceeds max")
        _num(c["max_abs_diff"], f"{where}.max_abs_diff", non_negative=True)
        _num(c["max_rel_diff"], f"{where}.max_rel_diff", non_negative=True)
        _sha256(c["ref_output_sha256"], f"{where}.ref_output_sha256")
        _sha256(c["online_output_sha256"], f"{where}.online_output_sha256")
    expected_pa4 = {(t, op) for t in (16, 32, 64) for op in ("forward", "backward", "joint")}
    if not expected_pa4.issubset(seen_pa4_t_op):
        _fail(f"$.payload.pa4_wider.cases: missing T/op coverage: {expected_pa4 - seen_pa4_t_op}")

    # 3. MP4
    mp4 = _obj(payload["mp4"], "$.payload.mp4")
    _keys(mp4, {"cases", "loss_scaler_verified", "provider"}, "$.payload.mp4")
    if mp4["provider"] != "mp4.cpu-f32.serial":
        _fail("$.payload.mp4.provider: unexpected provider")
    if not _bool(mp4["loss_scaler_verified"], "$.payload.mp4.loss_scaler_verified"):
        _fail("$.payload.mp4.loss_scaler_verified: must be true")
    mp4_cases = _arr(mp4["cases"], "$.payload.mp4.cases")
    seen_mp4_ops: set[str] = set()
    for i, raw_case in enumerate(mp4_cases):
        where = f"$.payload.mp4.cases[{i}]"
        c = _obj(raw_case, where)
        _keys(
            c,
            {
                "capability",
                "element_count",
                "iterations",
                "max_ns",
                "median_ns",
                "median_ns_per_iter",
                "min_ns",
                "narrow_bytes",
                "narrow_dtype",
                "operation",
                "output_sha256",
                "samples",
                "shape",
                "storage_byte_ratio",
                "storage_ratio_denominator",
                "storage_ratio_numerator",
                "throughput_melem_per_s",
                "warmup",
                "wide_bytes",
            },
            where,
        )
        op = _str(c["operation"], f"{where}.operation")
        seen_mp4_ops.add(op)
        _str(c["capability"], f"{where}.capability")
        if c["narrow_dtype"] not in {"f16", "bf16"}:
            _fail(f"{where}.narrow_dtype: expected f16 or bf16")
        elem = _int(c["element_count"], f"{where}.element_count", 1)
        narrow_b = _int(c["narrow_bytes"], f"{where}.narrow_bytes", 2)
        wide_b = _int(c["wide_bytes"], f"{where}.wide_bytes", 4)
        if narrow_b != 2 * elem or wide_b != 4 * elem or wide_b != 2 * narrow_b:
            _fail(f"{where}: exact 2:1 MP4 storage byte ratio violated ({wide_b} vs {narrow_b})")
        if _int(c["storage_ratio_numerator"], f"{where}.storage_ratio_numerator", 1) != 2:
            _fail(f"{where}.storage_ratio_numerator: must be 2")
        if _int(c["storage_ratio_denominator"], f"{where}.storage_ratio_denominator", 1) != 1:
            _fail(f"{where}.storage_ratio_denominator: must be 1")
        if _num(c["storage_byte_ratio"], f"{where}.storage_byte_ratio", strictly_positive=True) != 2.0:
            _fail(f"{where}.storage_byte_ratio: must be exactly 2.0")
        _shape(c["shape"], f"{where}.shape")
        _int(c["iterations"], f"{where}.iterations", 1)
        _int(c["samples"], f"{where}.samples", 1)
        _int(c["warmup"], f"{where}.warmup", 1)
        _validate_timing_triplet(c, where)
        _num(c["throughput_melem_per_s"], f"{where}.throughput_melem_per_s", strictly_positive=True)
        _sha256(c["output_sha256"], f"{where}.output_sha256")
    expected_mp4_ops = {
        "cast.f32-to-f16",
        "cast.f16-to-f32",
        "cast.f32-to-bf16",
        "cast.bf16-to-f32",
        "grad.unscale-f16-to-f32",
        "grad.unscale-bf16-to-f32",
    }
    if not expected_mp4_ops.issubset(seen_mp4_ops):
        _fail(f"$.payload.mp4.cases: missing operations {expected_mp4_ops - seen_mp4_ops}")

    # 4. AC4
    ac4 = _obj(payload["ac4"], "$.payload.ac4")
    _keys(ac4, {"base_shape", "cases", "dropout_rate"}, "$.payload.ac4")
    base_shape = _obj(ac4["base_shape"], "$.payload.ac4.base_shape")
    _keys(base_shape, {"batch", "head_dim", "kv_heads", "query_heads", "tokens"}, "$.payload.ac4.base_shape")
    for k in ("batch", "head_dim", "kv_heads", "query_heads", "tokens"):
        _int(base_shape[k], f"$.payload.ac4.base_shape.{k}", 1)
    _num(ac4["dropout_rate"], "$.payload.ac4.dropout_rate", non_negative=True)
    ac4_cases = _arr(ac4["cases"], "$.payload.ac4.cases")
    ac4_peaks: dict[tuple[int, str], int] = {}
    for i, raw_case in enumerate(ac4_cases):
        where = f"$.payload.ac4.cases[{i}]"
        c = _obj(raw_case, where)
        _keys(
            c,
            {
                "baseline_none_peak_bytes",
                "checkpoint_every",
                "gradient_staging_bytes",
                "iterations",
                "layers",
                "live_activation_peak_bytes",
                "max_ns",
                "median_ns",
                "median_ns_per_iter",
                "memory_reduction_pct",
                "min_ns",
                "output_sha256",
                "parity_with_none_bitwise",
                "planned_activation_peak_bytes",
                "policy",
                "rematerialized_layers",
                "retained_checkpoints",
                "retained_tapes",
                "samples",
                "saved_activation_bytes",
                "time_vs_none_ratio",
                "warmup",
                "workspace_bytes",
            },
            where,
        )
        layers = _int(c["layers"], f"{where}.layers", 1)
        policy = _str(c["policy"], f"{where}.policy")
        if policy not in {"NONE", "EVERY_LAYER", "SQRT_SEGMENTS", "EVERY_K"}:
            _fail(f"{where}.policy: invalid policy {policy!r}")
        _int(c["checkpoint_every"], f"{where}.checkpoint_every", 0)
        planned = _int(c["planned_activation_peak_bytes"], f"{where}.planned_activation_peak_bytes", 1)
        live = _int(c["live_activation_peak_bytes"], f"{where}.live_activation_peak_bytes", 1)
        base = _int(c["baseline_none_peak_bytes"], f"{where}.baseline_none_peak_bytes", 1)
        saved = _int(c["saved_activation_bytes"], f"{where}.saved_activation_bytes", 0)
        staging = _int(c["gradient_staging_bytes"], f"{where}.gradient_staging_bytes", 1)
        workspace = _int(c["workspace_bytes"], f"{where}.workspace_bytes", 1)
        if planned != live:
            _fail(f"{where}: live peak ({live}) != planned peak ({planned})")
        if saved != base - live:
            _fail(f"{where}: saved_activation_bytes mismatch")
        if workspace != live + staging:
            _fail(f"{where}: workspace_bytes mismatch")
        red = _num(c["memory_reduction_pct"], f"{where}.memory_reduction_pct", non_negative=True)
        expected_red = 100.0 * (base - live) / base
        if abs(red - expected_red) > 1e-4:
            _fail(f"{where}.memory_reduction_pct: {red} vs expected {expected_red}")
        if not _bool(c["parity_with_none_bitwise"], f"{where}.parity_with_none_bitwise"):
            _fail(f"{where}.parity_with_none_bitwise: must be true")
        _int(c["rematerialized_layers"], f"{where}.rematerialized_layers", 0)
        _int(c["retained_checkpoints"], f"{where}.retained_checkpoints", 0)
        _int(c["retained_tapes"], f"{where}.retained_tapes", 0)
        _int(c["iterations"], f"{where}.iterations", 1)
        _int(c["samples"], f"{where}.samples", 1)
        _int(c["warmup"], f"{where}.warmup", 1)
        _validate_timing_triplet(c, where)
        _num(c["time_vs_none_ratio"], f"{where}.time_vs_none_ratio", strictly_positive=True)
        _sha256(c["output_sha256"], f"{where}.output_sha256")
        ac4_peaks[(layers, policy)] = live

    for depth in (4, 8, 16):
        for pol in ("NONE", "EVERY_LAYER", "SQRT_SEGMENTS", "EVERY_K"):
            if (depth, pol) not in ac4_peaks:
                _fail(f"$.payload.ac4.cases: missing (L={depth}, policy={pol})")
        none_peak = ac4_peaks[(depth, "NONE")]
        for pol in ("EVERY_LAYER", "SQRT_SEGMENTS", "EVERY_K"):
            if not (ac4_peaks[(depth, pol)] < none_peak):
                _fail(
                    f"$.payload.ac4.cases: L={depth} {pol} peak ({ac4_peaks[(depth, pol)]}) "
                    f"must be strictly less than NONE ({none_peak})"
                )

    # 5. DIST5
    dist5 = _obj(payload["dist5"], "$.payload.dist5")
    _keys(dist5, {"cases", "transport"}, "$.payload.dist5")
    if dist5["transport"] != "af_unix_sock_stream":
        _fail("$.payload.dist5.transport: expected 'af_unix_sock_stream'")
    dist5_cases = _arr(dist5["cases"], "$.payload.dist5.cases")
    seen_dist5: set[tuple[int, str]] = set()
    for i, raw_case in enumerate(dist5_cases):
        where = f"$.payload.dist5.cases[{i}]"
        c = _obj(raw_case, where)
        _keys(
            c,
            {
                "element_count",
                "iterations",
                "max_ns",
                "median_ns",
                "median_ns_per_iter",
                "min_ns",
                "operation",
                "output_sha256",
                "payload_bytes",
                "samples",
                "verified_bitwise",
                "warmup",
                "world_size",
            },
            where,
        )
        op = _str(c["operation"], f"{where}.operation")
        if op not in {"allreduce_f32", "sync_gradients_f32"}:
            _fail(f"{where}.operation: unexpected op {op!r}")
        world = _int(c["world_size"], f"{where}.world_size", 1)
        if world not in {1, 2, 4}:
            _fail(f"{where}.world_size: expected 1, 2, or 4")
        seen_dist5.add((world, op))
        elem = _int(c["element_count"], f"{where}.element_count", 1)
        pbytes = _int(c["payload_bytes"], f"{where}.payload_bytes", 4)
        if pbytes != 4 * elem:
            _fail(f"{where}.payload_bytes: expected {4 * elem}, got {pbytes}")
        if not _bool(c["verified_bitwise"], f"{where}.verified_bitwise"):
            _fail(f"{where}.verified_bitwise: must be true")
        _int(c["iterations"], f"{where}.iterations", 1)
        _int(c["samples"], f"{where}.samples", 1)
        _int(c["warmup"], f"{where}.warmup", 1)
        _validate_timing_triplet(c, where)
        _sha256(c["output_sha256"], f"{where}.output_sha256")
    expected_dist5 = {(w, op) for w in (1, 2, 4) for op in ("allreduce_f32", "sync_gradients_f32")}
    if not expected_dist5.issubset(seen_dist5):
        _fail(f"$.payload.dist5.cases: missing world/op coverage {expected_dist5 - seen_dist5}")

    return payload


def load_and_verify_report(source: bytes | bytearray | str | Path) -> dict[str, Any]:
    raw = Path(source).read_bytes() if isinstance(source, (str, Path)) else bytes(source)
    if len(raw) > MAX_DOCUMENT_BYTES:
        _fail(f"report exceeds {MAX_DOCUMENT_BYTES} bytes")
    if raw.startswith(b"\xef\xbb\xbf"):
        _fail("report must not contain a UTF-8 BOM")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise Wave45BenchmarkError("report is not valid UTF-8") from exc
    try:
        envelope = json.loads(
            text,
            object_pairs_hook=_pairs,
            parse_constant=_forbid_constant,
        )
    except json.JSONDecodeError as exc:
        raise Wave45BenchmarkError(f"invalid JSON: {exc.msg}") from exc
    envelope = _obj(envelope, "$")
    if raw != canonical_json(envelope) + b"\n":
        _fail("report bytes are not canonical UTF-8 JSON with single trailing LF")
    _keys(envelope, {"checksum", "format", "payload", "version"}, "$")
    checksum = _obj(envelope["checksum"], "$.checksum")
    _keys(checksum, {"algorithm", "digest"}, "$.checksum")
    if checksum["algorithm"] != "sha256":
        _fail(f"unsupported checksum algorithm {checksum['algorithm']!r}")
    expected_digest = _sha256(checksum["digest"], "$.checksum.digest")
    integrity = {
        "format": envelope["format"],
        "payload": envelope["payload"],
        "version": envelope["version"],
    }
    actual_digest = hashlib.sha256(canonical_json(integrity)).hexdigest()
    if not hmac.compare_digest(expected_digest, actual_digest):
        _fail("SHA-256 checksum mismatch over canonical payload")
    if envelope["format"] != FORMAT_NAME:
        _fail(f"unsupported benchmark format {envelope['format']!r}")
    if _int(envelope["version"], "$.version", 1) != FORMAT_VERSION:
        _fail(f"unsupported benchmark version {envelope['version']!r}")
    return validate_payload(envelope["payload"])


def encode_report(
    payload: Mapping[str, Any],
    *,
    format_name: str = FORMAT_NAME,
    version: int = FORMAT_VERSION,
) -> bytes:
    integrity = {"format": format_name, "payload": payload, "version": version}
    digest = hashlib.sha256(canonical_json(integrity)).hexdigest()
    envelope = {
        "checksum": {"algorithm": "sha256", "digest": digest},
        "format": format_name,
        "payload": payload,
        "version": version,
    }
    return canonical_json(envelope) + b"\n"


def _default_report_path() -> Path:
    env_path = os.environ.get("WAVE45_BENCHMARK_REPORT")
    if env_path:
        return Path(env_path)
    return Path(__file__).resolve().parents[2] / "build" / "benchmarks" / "wave45-bench-v1-report.json"


class Wave45BenchmarkTests(unittest.TestCase):
    def _load_raw(self) -> bytes:
        path = _default_report_path()
        self.assertTrue(path.is_file(), f"expected generated benchmark report at {path}")
        return path.read_bytes()

    def test_emitted_report_passes_full_validation(self) -> None:
        raw = self._load_raw()
        payload = load_and_verify_report(raw)
        self.assertGreater(payload["process"]["self_peak_rss_kib"], 0)
        for c in payload["mp4"]["cases"]:
            self.assertEqual(c["wide_bytes"], 2 * c["narrow_bytes"])
            self.assertEqual(c["storage_byte_ratio"], 2.0)
        by_depth: dict[int, dict[str, int]] = {}
        for c in payload["ac4"]["cases"]:
            by_depth.setdefault(c["layers"], {})[c["policy"]] = c["live_activation_peak_bytes"]
        for depth in (4, 8, 16):
            self.assertLess(by_depth[depth]["EVERY_K"], by_depth[depth]["SQRT_SEGMENTS"])
            self.assertLess(by_depth[depth]["SQRT_SEGMENTS"], by_depth[depth]["EVERY_LAYER"])
            self.assertLess(by_depth[depth]["EVERY_LAYER"], by_depth[depth]["NONE"])

    def test_tampered_metric_value_fails_checksum(self) -> None:
        env = json.loads(self._load_raw().decode("utf-8"))
        env["payload"]["mod5"]["cases"][0]["median_ns"] += 1
        tampered = canonical_json(env) + b"\n"
        with self.assertRaisesRegex(Wave45BenchmarkError, "checksum mismatch"):
            load_and_verify_report(tampered)

    def test_missing_field_fails_schema(self) -> None:
        env = json.loads(self._load_raw().decode("utf-8"))
        payload = copy.deepcopy(env["payload"])
        del payload["process"]["self_peak_rss_kib"]
        reencoded = encode_report(payload)
        with self.assertRaisesRegex(Wave45BenchmarkError, "keys mismatch"):
            load_and_verify_report(reencoded)

    def test_wrong_version_fails_validator(self) -> None:
        env = json.loads(self._load_raw().decode("utf-8"))
        bad_ver = encode_report(env["payload"], version=2)
        with self.assertRaisesRegex(Wave45BenchmarkError, "unsupported benchmark version"):
            load_and_verify_report(bad_ver)

    def test_wrong_format_fails_validator(self) -> None:
        env = json.loads(self._load_raw().decode("utf-8"))
        bad_fmt = encode_report(env["payload"], format_name="wave45-bench-v2")
        with self.assertRaisesRegex(Wave45BenchmarkError, "unsupported benchmark format"):
            load_and_verify_report(bad_fmt)

    def test_non_finite_json_constant_rejected(self) -> None:
        raw = self._load_raw().decode("utf-8")
        # Inject NaN and Infinity into raw JSON bytes
        for bad_const in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(constant=bad_const):
                mutated = raw.replace('"storage_byte_ratio":2.0', f'"storage_byte_ratio":{bad_const}', 1)
                with self.assertRaisesRegex(Wave45BenchmarkError, "non-finite"):
                    load_and_verify_report(mutated.encode("utf-8"))

    def test_non_finite_python_float_rejected_by_validator(self) -> None:
        env = json.loads(self._load_raw().decode("utf-8"))
        for bad_val in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=bad_val):
                payload = copy.deepcopy(env["payload"])
                payload["mod5"]["cases"][0]["median_ns_per_iter"] = bad_val
                with self.assertRaisesRegex(Wave45BenchmarkError, "finite"):
                    validate_payload(payload)

    def test_non_positive_latency_rejected(self) -> None:
        env = json.loads(self._load_raw().decode("utf-8"))
        payload = copy.deepcopy(env["payload"])
        payload["dist5"]["cases"][0]["median_ns_per_iter"] = 0.0
        with self.assertRaisesRegex(Wave45BenchmarkError, "strictly positive"):
            load_and_verify_report(encode_report(payload))

    def test_mp4_broken_ratio_rejected(self) -> None:
        env = json.loads(self._load_raw().decode("utf-8"))
        payload = copy.deepcopy(env["payload"])
        payload["mp4"]["cases"][0]["narrow_bytes"] = payload["mp4"]["cases"][0]["wide_bytes"]
        with self.assertRaisesRegex(Wave45BenchmarkError, "2:1"):
            load_and_verify_report(encode_report(payload))

    def test_ac4_no_memory_reduction_rejected(self) -> None:
        env = json.loads(self._load_raw().decode("utf-8"))
        payload = copy.deepcopy(env["payload"])
        for c in payload["ac4"]["cases"]:
            if c["layers"] == 4 and c["policy"] == "EVERY_LAYER":
                none_peak = c["baseline_none_peak_bytes"]
                c["planned_activation_peak_bytes"] = none_peak
                c["live_activation_peak_bytes"] = none_peak
                c["saved_activation_bytes"] = 0
                c["memory_reduction_pct"] = 0.0
                c["workspace_bytes"] = none_peak + c["gradient_staging_bytes"]
                break
        with self.assertRaisesRegex(Wave45BenchmarkError, "strictly less than NONE"):
            load_and_verify_report(encode_report(payload))

    def test_non_canonical_whitespace_rejected(self) -> None:
        raw = self._load_raw().decode("utf-8")
        pretty = json.dumps(json.loads(raw), indent=2, sort_keys=True) + "\n"
        with self.assertRaisesRegex(Wave45BenchmarkError, "not canonical"):
            load_and_verify_report(pretty.encode("utf-8"))


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "--verify":
        payload = load_and_verify_report(argv[2])
        print(
            f"wave45-bench-v1 VERIFY PASS: mode={payload['toolchain']['build_mode']} "
            f"smoke={payload['measurement']['smoke_mode']} "
            f"mod5={len(payload['mod5']['cases'])} pa4={len(payload['pa4_wider']['cases'])} "
            f"mp4={len(payload['mp4']['cases'])} ac4={len(payload['ac4']['cases'])} "
            f"dist5={len(payload['dist5']['cases'])} "
            f"rss_kib={payload['process']['self_peak_rss_kib']}"
        )
        return 0
    return 1


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--verify":
        raise SystemExit(main(sys.argv))
    unittest.main()
