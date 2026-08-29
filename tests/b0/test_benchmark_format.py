from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tests.b0.benchmark_format import (
    CANONICAL_UPSTREAM_COMMIT,
    CANONICAL_UPSTREAM_REPOSITORY,
    DEFINITION_FORMAT,
    EMPTY_SHA256,
    REPORT_FORMAT,
    BenchmarkFormatError,
    canonical_json,
    definition_sha256,
    encode_definition,
    encode_report,
    load_definition,
    load_report,
    validate_definition_payload,
    validate_report_against_definition,
    validate_report_payload,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
PROJECT_COMMIT = "1" * 40


def _definition() -> dict[str, object]:
    return {
        "argv": ["build/eshkol-transformer-smoke"],
        "backend": {
            "acceleration": False,
            "device": "cpu",
            "evidence": "direct-process-execution",
            "name": "host-cpu",
        },
        "cwd": ".",
        "dtype": "not-applicable",
        "expected": {
            "exit_code": 0,
            "stderr": "empty",
            "stdout_sha256": HASH_A,
        },
        "id": "smoke-v1",
        "measurement": {
            "clock": "clock_monotonic",
            "completion_source": "linux_pidfd_poll",
            "elapsed_unit": "nanosecond",
            "memory_source": "linux_wait4_ru_maxrss",
            "peak_rss_unit": "kibibyte",
            "scope": "direct-child-process",
        },
        "repetitions": 2,
        "shapes": [],
        "timeout_ms": 30_000,
        "warmup": 1,
        "work": {"count": 1, "unit": "run"},
    }


def _report(definition_digest: str | None = None) -> dict[str, object]:
    if definition_digest is None:
        definition_digest = definition_sha256(encode_definition(_definition()))
    return {
        "stable": {
            "benchmark_id": "smoke-v1",
            "contract": {
                "backend": {
                    "acceleration": False,
                    "device": "cpu",
                    "evidence": "direct-process-execution",
                    "name": "host-cpu",
                },
                "dtype": "not-applicable",
                "repetitions": 2,
                "shapes": [],
                "warmup": 1,
                "work": {"count": 1, "unit": "run"},
            },
            "definition_sha256": definition_digest,
            "measurement_tool": {
                "binary_sha256": HASH_B,
                "source_path": "benchmarks/measure_linux.c",
                "source_sha256": HASH_C,
            },
            "outcome": {
                "exit_code": 0,
                "stderr_sha256": EMPTY_SHA256,
                "stdout_sha256": HASH_A,
            },
            "project": {"commit": PROJECT_COMMIT, "dirty": False},
            "target": {
                "path": "build/eshkol-transformer-smoke",
                "sha256": HASH_B,
            },
            "toolchain": {
                "binary_sha256": HASH_C,
                "commit": CANONICAL_UPSTREAM_COMMIT,
                "repository": CANONICAL_UPSTREAM_REPOSITORY,
                "version": "Eshkol Compiler v1.3.4-evolve",
            },
        },
        "volatile": {
            "environment": {
                "architecture": "x86_64",
                "compiler": "clang version 21.1.8",
                "cpu_model": "Example CPU",
                "kernel": "6.17.0",
                "logical_cpus": 8,
                "os_release": "Ubuntu 22.04",
            },
            "execution": {
                "samples": [
                    {
                        "elapsed_ns": 100,
                        "peak_rss_kib": 1000,
                        "repetition": 0,
                        "throughput_per_second": "10000000.000000",
                        "work_count": 1,
                    },
                    {
                        "elapsed_ns": 200,
                        "peak_rss_kib": 1001,
                        "repetition": 1,
                        "throughput_per_second": "5000000.000000",
                        "work_count": 1,
                    },
                ],
                "summary": {
                    "latency_ns_max": 200,
                    "latency_ns_median": 100,
                    "latency_ns_min": 100,
                    "peak_rss_kib": 1001,
                    "throughput_per_second": "6666666.666667",
                },
            },
            "observed_at_utc": "2026-08-28T12:34:56Z",
            "support_status": "supported",
        },
    }


def _document(encoded: bytes) -> dict[str, object]:
    return json.loads(encoded.decode("utf-8"))


def _resign(document: dict[str, object]) -> bytes:
    integrity = {
        "format": document["format"],
        "payload": document["payload"],
        "version": document["version"],
    }
    document["checksum"] = {
        "algorithm": "sha256",
        "digest": hashlib.sha256(canonical_json(integrity)).hexdigest(),
    }
    return canonical_json(document) + b"\n"


class CanonicalEnvelopeTests(unittest.TestCase):
    def test_definition_and_report_round_trip(self) -> None:
        definition = _definition()
        definition_bytes = encode_definition(definition)
        self.assertEqual(load_definition(definition_bytes), definition)
        self.assertEqual(
            definition_sha256(definition_bytes),
            hashlib.sha256(definition_bytes).hexdigest(),
        )

        report = _report(definition_sha256(definition_bytes))
        report_bytes = encode_report(report)
        self.assertEqual(load_report(report_bytes), report)
        validate_report_against_definition(
            load_report(report_bytes), definition, definition_sha256(definition_bytes)
        )
        self.assertTrue(definition_bytes.endswith(b"\n"))
        self.assertFalse(definition_bytes.endswith(b"\n\n"))
        self.assertTrue(report_bytes.endswith(b"\n"))

    def test_path_sources_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "definition.json"
            path.write_bytes(encode_definition(_definition()))
            self.assertEqual(load_definition(path), _definition())
            self.assertEqual(
                definition_sha256(path), hashlib.sha256(path.read_bytes()).hexdigest()
            )

    def test_duplicate_keys_noncanonical_bytes_bom_and_utf8_are_rejected(self) -> None:
        raw = encode_definition(_definition())
        duplicate = raw.replace(
            b'{"checksum":',
            b'{"format":"eshkol-benchmark-definition","checksum":',
            1,
        )
        with self.assertRaisesRegex(BenchmarkFormatError, "duplicate object key"):
            load_definition(duplicate)
        for malformed in (raw.replace(b",", b", ", 1), raw + b"\n", raw[:-1]):
            with self.subTest(malformed=malformed[-20:]):
                with self.assertRaisesRegex(BenchmarkFormatError, "not canonical"):
                    load_definition(malformed)
        with self.assertRaisesRegex(BenchmarkFormatError, "BOM"):
            load_definition(b"\xef\xbb\xbf" + raw)
        with self.assertRaisesRegex(BenchmarkFormatError, "not valid UTF-8"):
            load_definition(b"\xff\n")

    def test_checksum_corruption_and_malformed_checksum_are_rejected(self) -> None:
        document = _document(encode_definition(_definition()))
        document["payload"]["warmup"] = 9
        with self.assertRaisesRegex(BenchmarkFormatError, "checksum mismatch"):
            load_definition(canonical_json(document) + b"\n")

        for digest in ("0" * 64, "A" * 64, "0" * 63, True):
            with self.subTest(digest=digest):
                document = _document(encode_definition(_definition()))
                document["checksum"]["digest"] = digest
                message = "checksum mismatch" if digest == "0" * 64 else "invalid SHA-256|expected string"
                with self.assertRaisesRegex(BenchmarkFormatError, message):
                    load_definition(canonical_json(document) + b"\n")

    def test_unsupported_format_version_and_algorithm_are_rejected(self) -> None:
        document = _document(encode_definition(_definition()))
        document["format"] = REPORT_FORMAT
        with self.assertRaisesRegex(BenchmarkFormatError, "unsupported benchmark format"):
            load_definition(_resign(document))

        for version in (2, True):
            with self.subTest(version=version):
                document = _document(encode_definition(_definition()))
                document["version"] = version
                message = "unsupported benchmark version" if version == 2 else "expected integer"
                with self.assertRaisesRegex(BenchmarkFormatError, message):
                    load_definition(_resign(document))

        document = _document(encode_definition(_definition()))
        document["checksum"]["algorithm"] = "sha512"
        with self.assertRaisesRegex(BenchmarkFormatError, "unsupported checksum algorithm"):
            load_definition(canonical_json(document) + b"\n")

    def test_wrong_envelope_kind_is_rejected(self) -> None:
        with self.assertRaisesRegex(BenchmarkFormatError, "unsupported benchmark format"):
            load_report(encode_definition(_definition()))
        with self.assertRaisesRegex(BenchmarkFormatError, "unsupported benchmark format"):
            load_definition(encode_report(_report()))

    def test_non_json_values_and_lone_surrogates_are_rejected(self) -> None:
        for value, message in (
            ({"number": 1.5}, "floating-point"),
            ({"number": 2**70}, "signed 64-bit"),
            ({"value": object()}, "unsupported JSON type"),
            ({"text": "\ud800"}, "Unicode scalar text"),
        ):
            with self.subTest(value=repr(value)):
                with self.assertRaisesRegex(BenchmarkFormatError, message):
                    canonical_json(value)


class DefinitionContractTests(unittest.TestCase):
    def _reject(self, mutation: object, message: str) -> None:
        definition = _definition()
        mutation(definition)
        with self.assertRaisesRegex(BenchmarkFormatError, message):
            validate_definition_payload(definition)

    def test_missing_unknown_and_wrong_type_fields_are_rejected(self) -> None:
        self._reject(lambda value: value.pop("work"), "missing=.*work")
        self._reject(lambda value: value.update({"unknown": 1}), "unknown=.*unknown")
        self._reject(lambda value: value.update({"argv": "binary"}), "expected array")
        self._reject(lambda value: value.update({"backend": []}), "expected object")
        self._reject(lambda value: value.update({"id": 1}), "expected string")

    def test_boolean_values_are_not_accepted_as_integers(self) -> None:
        for field in ("warmup", "repetitions", "timeout_ms"):
            with self.subTest(field=field):
                self._reject(
                    lambda value, field=field: value.update({field: True}),
                    "expected integer",
                )
        self._reject(
            lambda value: value["work"].update({"count": False}), "expected integer"
        )
        self._reject(
            lambda value: value["expected"].update({"exit_code": True}),
            "expected integer",
        )

    def test_cpu_only_backend_constraints_are_explicit(self) -> None:
        mutations = (
            ("name", "gpu", "only 'host-cpu'"),
            ("device", "cuda", "only explicit cpu"),
            ("acceleration", True, "acceleration is unsupported"),
            ("acceleration", 0, "expected boolean"),
            ("evidence", "inferred", "direct-process-execution"),
        )
        for field, replacement, message in mutations:
            with self.subTest(field=field, replacement=replacement):
                self._reject(
                    lambda value, field=field, replacement=replacement: value[
                        "backend"
                    ].update({field: replacement}),
                    message,
                )

    def test_dtype_shapes_and_smoke_identity_are_fixed(self) -> None:
        self._reject(lambda value: value.update({"id": "training-v1"}), "smoke-v1")
        self._reject(lambda value: value.update({"dtype": "float32"}), "not-applicable")
        self._reject(lambda value: value.update({"shapes": [[1, 2]]}), "empty shape")
        self._reject(lambda value: value.update({"cwd": "build"}), "repository-root")

    def test_argument_vector_and_relative_executable_are_enforced(self) -> None:
        for argv, message in (
            ([], "non-empty"),
            (["/bin/true"], "relative"),
            (["../binary"], "relative"),
            (["."], "file path"),
            (["binary", 1], "expected string"),
            (["binary", "bad\x00argument"], "NUL"),
            (["some-other-program"], "canonical F0 smoke artifact"),
            (["build/eshkol-transformer-smoke", "extra"], "canonical F0 smoke artifact"),
        ):
            with self.subTest(argv=argv):
                self._reject(lambda value, argv=argv: value.update({"argv": argv}), message)

    def test_bounds_work_and_expected_output_are_enforced(self) -> None:
        for field, replacement, message in (
            ("repetitions", 0, "outside"),
            ("timeout_ms", 0, "outside"),
            ("timeout_ms", 3_600_001, "outside"),
        ):
            with self.subTest(field=field):
                self._reject(
                    lambda value, field=field, replacement=replacement: value.update(
                        {field: replacement}
                    ),
                    message,
                )
        self._reject(lambda value: value["work"].update({"count": 0}), "outside")
        self._reject(lambda value: value["work"].update({"unit": "token"}), "unit")
        self._reject(
            lambda value: value["expected"].update({"exit_code": 1}), "requires zero"
        )
        self._reject(
            lambda value: value["expected"].update({"stdout_sha256": "bad"}),
            "invalid SHA-256",
        )
        self._reject(
            lambda value: value["expected"].update({"stderr": "ignored"}),
            "requires 'empty'",
        )

    def test_measurement_fallbacks_and_unavailable_sources_are_rejected(self) -> None:
        cases = (
            ("clock", "time", "clock_monotonic"),
            ("completion_source", "polling", "linux_pidfd_poll"),
            ("elapsed_unit", "millisecond", "nanosecond"),
            ("memory_source", "unavailable", "linux_wait4_ru_maxrss"),
            ("peak_rss_unit", "byte", "kibibyte"),
            ("scope", "runner-and-child", "direct-child-process"),
        )
        for field, replacement, message in cases:
            with self.subTest(field=field):
                self._reject(
                    lambda value, field=field, replacement=replacement: value[
                        "measurement"
                    ].update({field: replacement}),
                    message,
                )


class ReportContractTests(unittest.TestCase):
    def _reject(self, mutation: object, message: str) -> None:
        report = _report()
        mutation(report)
        with self.assertRaisesRegex(BenchmarkFormatError, message):
            validate_report_payload(report)

    def test_stable_and_volatile_sections_are_exact(self) -> None:
        self._reject(lambda value: value.pop("stable"), "missing=.*stable")
        self._reject(
            lambda value: value["stable"].update({"hostname": "volatile"}),
            "unknown=.*hostname",
        )
        self._reject(
            lambda value: value["volatile"].update({"commit": PROJECT_COMMIT}),
            "unknown=.*commit",
        )
        self._reject(
            lambda value: value["stable"].update({"definition_sha256": "0"}),
            "invalid SHA-256",
        )

    def test_project_target_and_canonical_toolchain_are_enforced(self) -> None:
        self._reject(
            lambda value: value["stable"]["project"].update({"dirty": 0}),
            "expected boolean",
        )
        self._reject(
            lambda value: value["stable"]["project"].update({"commit": "main"}),
            "full lowercase Git commit",
        )
        self._reject(
            lambda value: value["stable"]["target"].update({"path": "/tmp/a"}),
            "relative",
        )
        self._reject(
            lambda value: value["stable"]["measurement_tool"].update(
                {"source_path": "benchmarks/fallback.c"}
            ),
            "unsupported measurement implementation",
        )
        self._reject(
            lambda value: value["stable"]["toolchain"].update(
                {"repository": "https://github.com/example/fork.git"}
            ),
            "unsupported upstream repository",
        )
        self._reject(
            lambda value: value["stable"]["toolchain"].update(
                {"commit": "2" * 40}
            ),
            "unsupported upstream commit",
        )

    def test_outcome_requires_success_and_explicit_hashes(self) -> None:
        self._reject(
            lambda value: value["stable"]["outcome"].update({"exit_code": 1}),
            "requires zero",
        )
        self._reject(
            lambda value: value["stable"]["outcome"].update(
                {"stderr_sha256": None}
            ),
            "expected string",
        )

    def test_environment_timestamp_and_support_status_are_validated(self) -> None:
        self._reject(
            lambda value: value["volatile"].update({"observed_at_utc": "yesterday"}),
            "ending in Z",
        )
        self._reject(
            lambda value: value["volatile"].update(
                {"observed_at_utc": "not-a-timeZ"}
            ),
            "canonical UTC timestamp",
        )
        self._reject(
            lambda value: value["volatile"].update(
                {"observed_at_utc": "2026-08-28Z"}
            ),
            "canonical UTC timestamp",
        )
        self._reject(
            lambda value: value["volatile"].update({"support_status": "unknown"}),
            "unsupported status",
        )
        self._reject(
            lambda value: value["volatile"]["environment"].update(
                {"logical_cpus": True}
            ),
            "expected integer",
        )
        self._reject(
            lambda value: value["volatile"]["environment"].update({"cpu_model": ""}),
            "non-empty",
        )

    def test_stable_contract_is_explicit_and_nonaccelerated(self) -> None:
        self._reject(
            lambda value: value["stable"]["contract"].update(
                {"dtype": "float64"}
            ),
            "not-applicable",
        )
        self._reject(
            lambda value: value["stable"]["contract"].update(
                {"shapes": [[1]]}
            ),
            "empty array",
        )
        self._reject(
            lambda value: value["stable"]["contract"]["backend"].update(
                {"acceleration": True}
            ),
            "acceleration is unsupported",
        )

    def test_sample_types_counts_and_order_are_enforced(self) -> None:
        self._reject(
            lambda value: value["stable"]["contract"].update({"repetitions": True}),
            "expected integer",
        )
        self._reject(
            lambda value: value["stable"]["contract"].update({"repetitions": 3}),
            "count must equal repetitions",
        )
        self._reject(
            lambda value: value["volatile"]["execution"]["samples"][1].update(
                {"repetition": 0}
            ),
            "consecutive and zero-based",
        )
        self._reject(
            lambda value: value["volatile"]["execution"]["samples"][0].update(
                {"elapsed_ns": 0}
            ),
            "outside",
        )
        self._reject(
            lambda value: value["volatile"]["execution"]["samples"][0].update(
                {"peak_rss_kib": "1000"}
            ),
            "expected integer",
        )
        self._reject(
            lambda value: value["volatile"]["execution"]["samples"][0].update(
                {"throughput_per_second": "1e6"}
            ),
            "plain decimal",
        )
        self._reject(
            lambda value: value["volatile"]["execution"]["samples"][0].update(
                {"throughput_per_second": "42"}
            ),
            "throughput mismatch",
        )

    def test_summary_order_and_values_are_enforced(self) -> None:
        self._reject(
            lambda value: value["volatile"]["execution"]["summary"].update(
                {"latency_ns_median": 300}
            ),
            "min <= median <= max",
        )
        self._reject(
            lambda value: value["volatile"]["execution"]["summary"].update(
                {"latency_ns_min": 99}
            ),
            "latency summary mismatch",
        )
        self._reject(
            lambda value: value["volatile"]["execution"]["summary"].update(
                {"peak_rss_kib": 9999}
            ),
            "peak RSS summary mismatch",
        )
        self._reject(
            lambda value: value["volatile"]["execution"]["summary"].update(
                {"throughput_per_second": "42"}
            ),
            "throughput summary mismatch",
        )

    def test_cross_document_mismatches_are_rejected(self) -> None:
        definition = _definition()
        encoded = encode_definition(definition)
        digest = definition_sha256(encoded)
        valid = _report(digest)
        validate_report_against_definition(valid, definition, digest)

        mutations = (
            (
                lambda value: value["stable"].update({"definition_sha256": HASH_B}),
                "definition_sha256",
            ),
            (
                lambda value: value["stable"]["target"].update({"path": "other"}),
                "target.path",
            ),
            (
                lambda value: value["stable"]["outcome"].update(
                    {"stdout_sha256": HASH_B}
                ),
                "outcome.stdout_sha256",
            ),
            (
                lambda value: value["stable"]["contract"].update({"warmup": 2}),
                "contract.warmup",
            ),
            (
                lambda value: value["volatile"]["execution"]["samples"][0].update(
                    {"work_count": 2}
                ),
                "throughput mismatch",
            ),
        )
        for mutation, message in mutations:
            with self.subTest(message=message):
                report = _report(digest)
                mutation(report)
                with self.assertRaisesRegex(BenchmarkFormatError, message):
                    validate_report_against_definition(report, definition, digest)


if __name__ == "__main__":
    unittest.main()
