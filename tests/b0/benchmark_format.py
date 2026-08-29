"""Strict, checksummed B0 benchmark definition and report formats (version 1)."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import unicodedata
from datetime import datetime
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, NoReturn

DEFINITION_FORMAT = "eshkol-benchmark-definition"
REPORT_FORMAT = "eshkol-benchmark-report"
FORMAT_VERSION = 1
MAX_DOCUMENT_BYTES = 16 * 1024 * 1024
MAX_JSON_DEPTH = 32
CANONICAL_UPSTREAM_REPOSITORY = "https://github.com/tsotchke/eshkol.git"
CANONICAL_UPSTREAM_COMMIT = "90cbd7130f47b8184bcc77b8d5c1b0026da980de"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_IDENTIFIER = re.compile(r"[a-z][a-z0-9-]{0,63}\Z")
_DECIMAL = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z")
_UTC_TIMESTAMP = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,6})?Z\Z"
)


class BenchmarkFormatError(ValueError):
    """Malformed, corrupt, noncanonical, or unsupported benchmark data."""


def _fail(message: str) -> NoReturn:
    raise BenchmarkFormatError(message)


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{where}: expected object")
    return value


def _array(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{where}: expected array")
    return value


def _string(value: Any, where: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str):
        _fail(f"{where}: expected string")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise BenchmarkFormatError(
            f"{where}: string is not valid Unicode scalar text"
        ) from error
    if unicodedata.normalize("NFC", value) != value:
        _fail(f"{where}: string is not Unicode NFC")
    if nonempty and not value:
        _fail(f"{where}: expected non-empty string")
    return value


def _integer(
    value: Any,
    where: str,
    minimum: int = 0,
    maximum: int = 2**63 - 1,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{where}: expected integer")
    if value < minimum or value > maximum:
        _fail(f"{where}: integer outside [{minimum}, {maximum}]")
    return value


def _keys(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    actual = set(value)
    if actual != expected:
        _fail(
            f"{where}: keys differ; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate object key: {key}")
        result[key] = value
    return result


def _forbid_float(token: str) -> NoReturn:
    _fail(f"JSON floating-point number is forbidden: {token}")


def _forbid_constant(token: str) -> NoReturn:
    _fail(f"non-standard JSON number is forbidden: {token}")


def _validate_json(value: Any, where: str = "$", depth: int = 0) -> None:
    if depth > MAX_JSON_DEPTH:
        _fail(f"{where}: JSON nesting exceeds {MAX_JSON_DEPTH}")
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        if value < -(2**63) or value > 2**63 - 1:
            _fail(f"{where}: integer outside signed 64-bit range")
        return
    if isinstance(value, float):
        _fail(f"{where}: JSON floating-point values are forbidden")
    if isinstance(value, str):
        _string(value, where)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json(item, f"{where}[{index}]", depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _string(key, f"{where}.<key>")
            _validate_json(item, f"{where}.{key}", depth + 1)
        return
    _fail(f"{where}: unsupported JSON type {type(value).__name__}")


def canonical_json(value: Any) -> bytes:
    """Return canonical UTF-8 JSON without its required document LF."""
    _validate_json(value)
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8", errors="strict")
    except (UnicodeEncodeError, UnicodeError) as error:
        raise BenchmarkFormatError(
            "benchmark data cannot be encoded as UTF-8"
        ) from error


def _sha256(value: Any, where: str) -> str:
    digest = _string(value, where)
    if _SHA256.fullmatch(digest) is None:
        _fail(f"{where}: invalid SHA-256")
    return digest


def _identifier(value: Any, where: str) -> str:
    identifier = _string(value, where)
    if _IDENTIFIER.fullmatch(identifier) is None:
        _fail(f"{where}: invalid identifier")
    return identifier


def _relative_path(value: Any, where: str) -> str:
    text = _string(value, where, nonempty=True)
    if "\x00" in text or "\n" in text or "\r" in text or "\\" in text:
        _fail(f"{where}: invalid relative POSIX path")
    path = PurePosixPath(text)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        _fail(f"{where}: expected path relative to repository root")
    if text in {".", ".."}:
        _fail(f"{where}: expected file path")
    return text


def _validate_backend(raw: Any, where: str) -> dict[str, Any]:
    backend = _object(raw, where)
    _keys(backend, {"acceleration", "device", "evidence", "name"}, where)
    if backend["name"] != "host-cpu":
        _fail(f"{where}.name: version 1 supports only 'host-cpu'")
    if backend["device"] != "cpu":
        _fail(f"{where}.device: version 1 supports only explicit cpu")
    if not isinstance(backend["acceleration"], bool):
        _fail(f"{where}.acceleration: expected boolean")
    if backend["acceleration"]:
        _fail(f"{where}.acceleration: acceleration is unsupported in version 1")
    if backend["evidence"] != "direct-process-execution":
        _fail(
            f"{where}.evidence: version 1 requires 'direct-process-execution'"
        )
    return backend


def _validate_definition_payload(raw: Any) -> dict[str, Any]:
    where = "$.payload"
    payload = _object(raw, where)
    _keys(
        payload,
        {
            "argv",
            "backend",
            "cwd",
            "dtype",
            "expected",
            "id",
            "measurement",
            "repetitions",
            "shapes",
            "timeout_ms",
            "warmup",
            "work",
        },
        where,
    )
    benchmark_id = _identifier(payload["id"], f"{where}.id")
    if benchmark_id != "smoke-v1":
        _fail(f"{where}.id: version 1 supports only 'smoke-v1'")

    argv = _array(payload["argv"], f"{where}.argv")
    if not argv:
        _fail(f"{where}.argv: expected a non-empty argument vector")
    _relative_path(argv[0], f"{where}.argv[0]")
    for index, value in enumerate(argv[1:], 1):
        argument = _string(value, f"{where}.argv[{index}]")
        if "\x00" in argument:
            _fail(f"{where}.argv[{index}]: NUL is forbidden")
    if argv != ["build/eshkol-transformer-smoke"]:
        _fail(
            f"{where}.argv: version 1 requires the canonical F0 smoke artifact "
            "with no arguments"
        )
    if payload["cwd"] != ".":
        _fail(f"{where}.cwd: version 1 requires repository-root '.'")

    _validate_backend(payload["backend"], f"{where}.backend")
    if payload["dtype"] != "not-applicable":
        _fail(f"{where}.dtype: smoke benchmark dtype must be 'not-applicable'")
    shapes = _array(payload["shapes"], f"{where}.shapes")
    if shapes:
        _fail(f"{where}.shapes: smoke benchmark requires an empty shape list")
    _integer(payload["warmup"], f"{where}.warmup")
    _integer(payload["repetitions"], f"{where}.repetitions", 1)
    _integer(payload["timeout_ms"], f"{where}.timeout_ms", 1, 3_600_000)

    work = _object(payload["work"], f"{where}.work")
    _keys(work, {"count", "unit"}, f"{where}.work")
    _integer(work["count"], f"{where}.work.count", 1)
    if work["unit"] != "run":
        _fail(f"{where}.work.unit: smoke benchmark work unit must be 'run'")

    expected = _object(payload["expected"], f"{where}.expected")
    _keys(expected, {"exit_code", "stderr", "stdout_sha256"}, f"{where}.expected")
    exit_code = _integer(expected["exit_code"], f"{where}.expected.exit_code")
    if exit_code != 0:
        _fail(f"{where}.expected.exit_code: smoke benchmark requires zero")
    _sha256(expected["stdout_sha256"], f"{where}.expected.stdout_sha256")
    if expected["stderr"] != "empty":
        _fail(f"{where}.expected.stderr: smoke benchmark requires 'empty'")

    measurement = _object(payload["measurement"], f"{where}.measurement")
    _keys(
        measurement,
        {
            "clock",
            "completion_source",
            "elapsed_unit",
            "memory_source",
            "peak_rss_unit",
            "scope",
        },
        f"{where}.measurement",
    )
    required_measurement = {
        "clock": "clock_monotonic",
        "completion_source": "linux_pidfd_poll",
        "elapsed_unit": "nanosecond",
        "memory_source": "linux_wait4_ru_maxrss",
        "peak_rss_unit": "kibibyte",
        "scope": "direct-child-process",
    }
    for field, required in required_measurement.items():
        if measurement[field] != required:
            _fail(
                f"{where}.measurement.{field}: version 1 requires {required!r}; "
                "fallback or unavailable measurements must be reported explicitly"
            )
    return payload


def _validate_project(raw: Any, where: str) -> dict[str, Any]:
    project = _object(raw, where)
    _keys(project, {"commit", "dirty"}, where)
    commit = _string(project["commit"], f"{where}.commit")
    if _COMMIT.fullmatch(commit) is None:
        _fail(f"{where}.commit: expected full lowercase Git commit")
    if not isinstance(project["dirty"], bool):
        _fail(f"{where}.dirty: expected boolean")
    return project


def _validate_toolchain(raw: Any, where: str) -> dict[str, Any]:
    toolchain = _object(raw, where)
    _keys(
        toolchain,
        {"binary_sha256", "commit", "repository", "version"},
        where,
    )
    if toolchain["repository"] != CANONICAL_UPSTREAM_REPOSITORY:
        _fail(f"{where}.repository: unsupported upstream repository")
    if toolchain["commit"] != CANONICAL_UPSTREAM_COMMIT:
        _fail(f"{where}.commit: unsupported upstream commit")
    version = _string(toolchain["version"], f"{where}.version", nonempty=True)
    if len(version) > 128:
        _fail(f"{where}.version: version text exceeds 128 characters")
    _sha256(toolchain["binary_sha256"], f"{where}.binary_sha256")
    return toolchain


def _validate_outcome(raw: Any, where: str) -> dict[str, Any]:
    outcome = _object(raw, where)
    _keys(outcome, {"exit_code", "stderr_sha256", "stdout_sha256"}, where)
    if _integer(outcome["exit_code"], f"{where}.exit_code") != 0:
        _fail(f"{where}.exit_code: a completed smoke report requires zero")
    _sha256(outcome["stdout_sha256"], f"{where}.stdout_sha256")
    _sha256(outcome["stderr_sha256"], f"{where}.stderr_sha256")
    return outcome


def _decimal(value: Any, where: str) -> str:
    text = _string(value, where)
    if _DECIMAL.fullmatch(text) is None:
        _fail(f"{where}: expected non-negative plain decimal string")
    return text


def decimal_rate(work_count: int, elapsed_ns: int) -> str:
    """Return the version-1 canonical runs-per-second decimal."""
    with localcontext() as context:
        context.prec = 40
        rate = Decimal(work_count) * Decimal(1_000_000_000) / Decimal(elapsed_ns)
        rounded = rate.quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN)
    return format(rounded, "f")


def _validate_sample(raw: Any, where: str) -> dict[str, Any]:
    sample = _object(raw, where)
    _keys(
        sample,
        {
            "elapsed_ns",
            "peak_rss_kib",
            "repetition",
            "throughput_per_second",
            "work_count",
        },
        where,
    )
    _integer(sample["repetition"], f"{where}.repetition")
    _integer(sample["elapsed_ns"], f"{where}.elapsed_ns", 1)
    _integer(sample["peak_rss_kib"], f"{where}.peak_rss_kib", 1)
    _integer(sample["work_count"], f"{where}.work_count", 1)
    throughput = _decimal(
        sample["throughput_per_second"], f"{where}.throughput_per_second"
    )
    if throughput != decimal_rate(sample["work_count"], sample["elapsed_ns"]):
        _fail(f"{where}.throughput_per_second: throughput mismatch")
    return sample


def _validate_summary(raw: Any, where: str) -> dict[str, Any]:
    summary = _object(raw, where)
    _keys(
        summary,
        {
            "latency_ns_max",
            "latency_ns_median",
            "latency_ns_min",
            "peak_rss_kib",
            "throughput_per_second",
        },
        where,
    )
    minimum = _integer(summary["latency_ns_min"], f"{where}.latency_ns_min", 1)
    median = _integer(
        summary["latency_ns_median"], f"{where}.latency_ns_median", 1
    )
    maximum = _integer(summary["latency_ns_max"], f"{where}.latency_ns_max", 1)
    if not minimum <= median <= maximum:
        _fail(f"{where}: latency requires min <= median <= max")
    _integer(summary["peak_rss_kib"], f"{where}.peak_rss_kib", 1)
    _decimal(summary["throughput_per_second"], f"{where}.throughput_per_second")
    return summary


def _validate_report_payload(raw: Any) -> dict[str, Any]:
    payload = _object(raw, "$.payload")
    _keys(payload, {"stable", "volatile"}, "$.payload")

    stable = _object(payload["stable"], "$.payload.stable")
    _keys(
        stable,
        {
            "benchmark_id",
            "contract",
            "definition_sha256",
            "measurement_tool",
            "outcome",
            "project",
            "target",
            "toolchain",
        },
        "$.payload.stable",
    )
    if _identifier(stable["benchmark_id"], "$.payload.stable.benchmark_id") != "smoke-v1":
        _fail("$.payload.stable.benchmark_id: version 1 supports only 'smoke-v1'")
    contract = _object(stable["contract"], "$.payload.stable.contract")
    _keys(
        contract,
        {"backend", "dtype", "repetitions", "shapes", "warmup", "work"},
        "$.payload.stable.contract",
    )
    _validate_backend(contract["backend"], "$.payload.stable.contract.backend")
    if contract["dtype"] != "not-applicable":
        _fail("$.payload.stable.contract.dtype: expected 'not-applicable'")
    shapes = _array(contract["shapes"], "$.payload.stable.contract.shapes")
    if shapes:
        _fail("$.payload.stable.contract.shapes: expected an empty array")
    _integer(contract["warmup"], "$.payload.stable.contract.warmup")
    repetitions = _integer(
        contract["repetitions"], "$.payload.stable.contract.repetitions", 1
    )
    work = _object(contract["work"], "$.payload.stable.contract.work")
    _keys(work, {"count", "unit"}, "$.payload.stable.contract.work")
    _integer(work["count"], "$.payload.stable.contract.work.count", 1)
    if work["unit"] != "run":
        _fail("$.payload.stable.contract.work.unit: expected 'run'")
    _sha256(stable["definition_sha256"], "$.payload.stable.definition_sha256")
    measurement_tool = _object(
        stable["measurement_tool"], "$.payload.stable.measurement_tool"
    )
    _keys(
        measurement_tool,
        {"binary_sha256", "source_path", "source_sha256"},
        "$.payload.stable.measurement_tool",
    )
    if measurement_tool["source_path"] != "benchmarks/measure_linux.c":
        _fail(
            "$.payload.stable.measurement_tool.source_path: unsupported "
            "measurement implementation"
        )
    _sha256(
        measurement_tool["source_sha256"],
        "$.payload.stable.measurement_tool.source_sha256",
    )
    _sha256(
        measurement_tool["binary_sha256"],
        "$.payload.stable.measurement_tool.binary_sha256",
    )
    _validate_project(stable["project"], "$.payload.stable.project")
    target = _object(stable["target"], "$.payload.stable.target")
    _keys(target, {"path", "sha256"}, "$.payload.stable.target")
    _relative_path(target["path"], "$.payload.stable.target.path")
    _sha256(target["sha256"], "$.payload.stable.target.sha256")
    _validate_toolchain(stable["toolchain"], "$.payload.stable.toolchain")
    _validate_outcome(stable["outcome"], "$.payload.stable.outcome")

    volatile = _object(payload["volatile"], "$.payload.volatile")
    _keys(
        volatile,
        {"environment", "execution", "observed_at_utc", "support_status"},
        "$.payload.volatile",
    )
    timestamp = _string(
        volatile["observed_at_utc"], "$.payload.volatile.observed_at_utc"
    )
    if _UTC_TIMESTAMP.fullmatch(timestamp) is None:
        _fail(
            "$.payload.volatile.observed_at_utc: expected canonical UTC timestamp "
            "ending in Z"
        )
    try:
        parsed = datetime.fromisoformat(timestamp[:-1] + "+00:00")
    except ValueError as error:
        raise BenchmarkFormatError(
            "$.payload.volatile.observed_at_utc: invalid ISO-8601 timestamp"
        ) from error
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        _fail("$.payload.volatile.observed_at_utc: expected UTC timestamp")
    if volatile["support_status"] not in {"supported", "compatibility-only"}:
        _fail("$.payload.volatile.support_status: unsupported status")

    environment = _object(
        volatile["environment"], "$.payload.volatile.environment"
    )
    _keys(
        environment,
        {
            "architecture",
            "compiler",
            "cpu_model",
            "kernel",
            "logical_cpus",
            "os_release",
        },
        "$.payload.volatile.environment",
    )
    for field in ("architecture", "compiler", "cpu_model", "kernel", "os_release"):
        _string(
            environment[field],
            f"$.payload.volatile.environment.{field}",
            nonempty=True,
        )
    _integer(
        environment["logical_cpus"],
        "$.payload.volatile.environment.logical_cpus",
        1,
    )

    execution = _object(volatile["execution"], "$.payload.volatile.execution")
    _keys(
        execution,
        {"samples", "summary"},
        "$.payload.volatile.execution",
    )
    samples = _array(execution["samples"], "$.payload.volatile.execution.samples")
    if len(samples) != repetitions:
        _fail("$.payload.volatile.execution.samples: count must equal repetitions")
    checked_samples = [
        _validate_sample(sample, f"$.payload.volatile.execution.samples[{index}]")
        for index, sample in enumerate(samples)
    ]
    indices = [sample["repetition"] for sample in checked_samples]
    if indices != list(range(repetitions)):
        _fail(
            "$.payload.volatile.execution.samples: repetitions must be consecutive "
            "and zero-based"
        )
    summary = _validate_summary(
        execution["summary"], "$.payload.volatile.execution.summary"
    )
    elapsed = sorted(sample["elapsed_ns"] for sample in checked_samples)
    expected_median = elapsed[(len(elapsed) - 1) // 2]
    if (
        summary["latency_ns_min"] != elapsed[0]
        or summary["latency_ns_median"] != expected_median
        or summary["latency_ns_max"] != elapsed[-1]
    ):
        _fail("$.payload.volatile.execution.summary: latency summary mismatch")
    if summary["peak_rss_kib"] != max(
        sample["peak_rss_kib"] for sample in checked_samples
    ):
        _fail("$.payload.volatile.execution.summary: peak RSS summary mismatch")
    total_work = sum(sample["work_count"] for sample in checked_samples)
    total_elapsed = sum(sample["elapsed_ns"] for sample in checked_samples)
    if summary["throughput_per_second"] != decimal_rate(total_work, total_elapsed):
        _fail("$.payload.volatile.execution.summary: throughput summary mismatch")
    return payload


def validate_definition_payload(raw: Any) -> dict[str, Any]:
    """Validate and return a version-1 definition payload."""
    return _validate_definition_payload(raw)


def validate_report_payload(raw: Any) -> dict[str, Any]:
    """Validate and return a version-1 report payload."""
    return _validate_report_payload(raw)


def _integrity_object(format_name: str, payload: Any) -> dict[str, Any]:
    return {"format": format_name, "payload": payload, "version": FORMAT_VERSION}


def _encode(format_name: str, payload: Mapping[str, Any], validator: Any) -> bytes:
    validated = validator(dict(payload))
    integrity = _integrity_object(format_name, validated)
    digest = hashlib.sha256(canonical_json(integrity)).hexdigest()
    envelope = {
        "checksum": {"algorithm": "sha256", "digest": digest},
        **integrity,
    }
    return canonical_json(envelope) + b"\n"


def encode_definition(payload: Mapping[str, Any]) -> bytes:
    """Validate and encode one canonical benchmark definition."""
    return _encode(DEFINITION_FORMAT, payload, _validate_definition_payload)


def encode_report(payload: Mapping[str, Any]) -> bytes:
    """Validate and encode one canonical generated benchmark report."""
    return _encode(REPORT_FORMAT, payload, _validate_report_payload)


def _source_bytes(source: bytes | bytearray | memoryview | str | Path) -> bytes:
    return Path(source).read_bytes() if isinstance(source, (str, Path)) else bytes(source)


def _load(
    source: bytes | bytearray | memoryview | str | Path,
    format_name: str,
    validator: Any,
) -> dict[str, Any]:
    raw = _source_bytes(source)
    if len(raw) > MAX_DOCUMENT_BYTES:
        _fail(f"benchmark document exceeds {MAX_DOCUMENT_BYTES} bytes")
    if raw.startswith(b"\xef\xbb\xbf"):
        _fail("benchmark document must not contain a UTF-8 BOM")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise BenchmarkFormatError("benchmark document is not valid UTF-8") from error
    try:
        envelope = json.loads(
            text,
            object_pairs_hook=_pairs,
            parse_float=_forbid_float,
            parse_constant=_forbid_constant,
        )
    except json.JSONDecodeError as error:
        raise BenchmarkFormatError(f"invalid JSON: {error.msg}") from error
    except RecursionError as error:
        raise BenchmarkFormatError(
            "benchmark document JSON nesting exceeds parser limit"
        ) from error
    envelope = _object(envelope, "$")
    if raw != canonical_json(envelope) + b"\n":
        _fail("benchmark document bytes are not canonical")
    _keys(envelope, {"checksum", "format", "payload", "version"}, "$")
    checksum = _object(envelope["checksum"], "$.checksum")
    _keys(checksum, {"algorithm", "digest"}, "$.checksum")
    algorithm = _string(checksum["algorithm"], "$.checksum.algorithm")
    if algorithm != "sha256":
        _fail(f"unsupported checksum algorithm {algorithm!r}")
    expected = _sha256(checksum["digest"], "$.checksum.digest")
    integrity = {
        "format": envelope["format"],
        "payload": envelope["payload"],
        "version": envelope["version"],
    }
    actual = hashlib.sha256(canonical_json(integrity)).hexdigest()
    if not hmac.compare_digest(expected, actual):
        _fail("benchmark document checksum mismatch")
    actual_format = _string(envelope["format"], "$.format")
    version = _integer(envelope["version"], "$.version", 1)
    if actual_format != format_name:
        _fail(f"unsupported benchmark format {actual_format!r}")
    if version != FORMAT_VERSION:
        _fail(f"unsupported benchmark version {version!r}")
    return validator(envelope["payload"])


def load_definition(
    source: bytes | bytearray | memoryview | str | Path,
) -> dict[str, Any]:
    """Load a canonical, checksummed benchmark definition."""
    return _load(source, DEFINITION_FORMAT, _validate_definition_payload)


def load_report(
    source: bytes | bytearray | memoryview | str | Path,
) -> dict[str, Any]:
    """Load a canonical, checksummed generated benchmark report."""
    return _load(source, REPORT_FORMAT, _validate_report_payload)


def definition_sha256(
    source: bytes | bytearray | memoryview | str | Path,
) -> str:
    """Return SHA-256 of validated canonical definition bytes, including its LF."""
    raw = _source_bytes(source)
    load_definition(raw)
    return hashlib.sha256(raw).hexdigest()


def validate_report_against_definition(
    report: Mapping[str, Any],
    definition: Mapping[str, Any],
    expected_definition_sha256: str,
) -> None:
    """Reject a valid report whose stable facts do not match its definition."""
    checked_report = _validate_report_payload(dict(report))
    checked_definition = _validate_definition_payload(dict(definition))
    _sha256(expected_definition_sha256, "expected_definition_sha256")
    stable = checked_report["stable"]
    contract = stable["contract"]
    execution = checked_report["volatile"]["execution"]
    mismatches: list[str] = []
    comparisons = (
        ("benchmark_id", stable["benchmark_id"], checked_definition["id"]),
        (
            "definition_sha256",
            stable["definition_sha256"],
            expected_definition_sha256,
        ),
        ("target.path", stable["target"]["path"], checked_definition["argv"][0]),
        (
            "outcome.exit_code",
            stable["outcome"]["exit_code"],
            checked_definition["expected"]["exit_code"],
        ),
        (
            "outcome.stdout_sha256",
            stable["outcome"]["stdout_sha256"],
            checked_definition["expected"]["stdout_sha256"],
        ),
        ("outcome.stderr_sha256", stable["outcome"]["stderr_sha256"], EMPTY_SHA256),
        ("contract.backend", contract["backend"], checked_definition["backend"]),
        ("contract.dtype", contract["dtype"], checked_definition["dtype"]),
        ("contract.shapes", contract["shapes"], checked_definition["shapes"]),
        ("contract.warmup", contract["warmup"], checked_definition["warmup"]),
        (
            "contract.repetitions",
            contract["repetitions"],
            checked_definition["repetitions"],
        ),
        ("contract.work", contract["work"], checked_definition["work"]),
    )
    for field, actual, expected in comparisons:
        if actual != expected:
            mismatches.append(field)
    if any(
        sample["work_count"] != contract["work"]["count"]
        for sample in execution["samples"]
    ):
        mismatches.append("execution.samples.work_count")
    if mismatches:
        _fail(
            "report does not match benchmark definition: " + ", ".join(mismatches)
        )
