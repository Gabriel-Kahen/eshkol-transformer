from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

try:
    from .manifest_format import (
        ManifestError,
        canonical_bytes,
        fingerprint,
        load_manifest_bytes,
    )
except ImportError:
    from manifest_format import (  # type: ignore[no-redef]
        ManifestError,
        canonical_bytes,
        fingerprint,
        load_manifest_bytes,
    )


FIXTURE = Path(__file__).with_name("fixtures") / "resolved_minimal_v1.json"
FINGERPRINT = Path(__file__).with_name("fixtures") / "resolved_minimal_v1.sha256"
SOURCE_FIXTURE = Path(__file__).with_name("fixtures") / "minimal_config_v1.json"


class ManifestFormatTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = FIXTURE.read_bytes()
        self.value = load_manifest_bytes(self.raw)

    def test_checked_fixture_is_canonical_and_fingerprinted(self) -> None:
        self.assertEqual(canonical_bytes(self.value), self.raw)
        self.assertEqual(FINGERPRINT.read_text(encoding="ascii").strip(), fingerprint(self.raw))

    def test_source_fixture_is_compact_flat_json(self) -> None:
        raw = SOURCE_FIXTURE.read_bytes()
        value = json.loads(raw)
        self.assertIsInstance(value, dict)
        self.assertEqual(len(value), 8)
        self.assertEqual(value["config-schema-major"], 1)
        self.assertEqual(value["config-schema-minor"], 0)
        self.assertEqual(canonical_bytes(value), raw)

    def test_payload_mutation_changes_fingerprint(self) -> None:
        changed = copy.deepcopy(self.value)
        changed["resolved"]["run.seed"] += 1
        self.assertNotEqual(fingerprint(canonical_bytes(changed)), fingerprint(self.raw))

    def test_provenance_is_part_of_identity(self) -> None:
        changed = copy.deepcopy(self.value)
        changed["provenance"]["model.device"] = "input"
        load_manifest_bytes(canonical_bytes(changed))
        self.assertNotEqual(fingerprint(canonical_bytes(changed)), fingerprint(self.raw))

    def test_duplicate_unknown_and_missing_keys_are_rejected(self) -> None:
        duplicate = self.raw.replace(b'{"canonicalization":', b'{"format":"x","canonicalization":', 1)
        with self.assertRaises(ManifestError):
            load_manifest_bytes(duplicate)
        unknown = copy.deepcopy(self.value)
        unknown["unknown"] = 1
        with self.assertRaises(ManifestError):
            load_manifest_bytes(canonical_bytes(unknown))
        missing = copy.deepcopy(self.value)
        del missing["limits"]
        with self.assertRaises(ManifestError):
            load_manifest_bytes(canonical_bytes(missing))

    def test_noncanonical_bytes_are_rejected(self) -> None:
        variants = [
            self.raw[:-1],
            self.raw + b"\n",
            self.raw.replace(b"\n", b"\r\n"),
            b"\xef\xbb\xbf" + self.raw,
            b" " + self.raw,
            json.dumps(self.value, indent=2, sort_keys=True).encode() + b"\n",
        ]
        for raw in variants:
            with self.subTest(raw=raw[:20]):
                with self.assertRaises(ManifestError):
                    load_manifest_bytes(raw)

    def test_versions_features_limits_and_provenance_are_exact(self) -> None:
        mutations = [
            ("format-version", [2, 0]),
            ("format-version", [True, 0]),
            ("config-schema-version", [1, 1]),
            ("required-features", ["future"]),
        ]
        for key, replacement in mutations:
            changed = copy.deepcopy(self.value)
            changed[key] = replacement
            with self.subTest(key=key, replacement=replacement):
                with self.assertRaises(ManifestError):
                    load_manifest_bytes(canonical_bytes(changed))
        changed = copy.deepcopy(self.value)
        changed["limits"]["input-bytes"] = 1
        with self.assertRaises(ManifestError):
            load_manifest_bytes(canonical_bytes(changed))
        changed = copy.deepcopy(self.value)
        changed["limits"]["max-input-nesting-depth"] = True
        with self.assertRaises(ManifestError):
            load_manifest_bytes(canonical_bytes(changed))
        changed = copy.deepcopy(self.value)
        changed["provenance"]["model.device"] = "environment"
        with self.assertRaises(ManifestError):
            load_manifest_bytes(canonical_bytes(changed))
        for replacement in [["default"], {"source": "default"}, True, 1]:
            changed = copy.deepcopy(self.value)
            changed["provenance"]["model.device"] = replacement
            with self.subTest(provenance_type=type(replacement).__name__):
                with self.assertRaises(ManifestError):
                    load_manifest_bytes(canonical_bytes(changed))
        changed = copy.deepcopy(self.value)
        changed["provenance"]["config-schema-major"] = "override"
        with self.assertRaises(ManifestError):
            load_manifest_bytes(canonical_bytes(changed))
        for key, replacement in [
            ("training.accumulation-steps", 2),
            ("model.kv-head-count", 2),
        ]:
            changed = copy.deepcopy(self.value)
            changed["resolved"][key] = replacement
            with self.subTest(default_provenance=key):
                with self.assertRaises(ManifestError):
                    load_manifest_bytes(canonical_bytes(changed))

    def test_invalid_types_ranges_and_combinations_are_rejected(self) -> None:
        for key, replacement in [
            ("model.device", "accelerator"),
            ("model.dtype", "f16"),
            ("run.deterministic", False),
            ("run.seed", -1),
            ("model.context-length", 0),
            ("config-schema-major", True),
            ("config-schema-minor", False),
        ]:
            changed = copy.deepcopy(self.value)
            changed["resolved"][key] = replacement
            with self.subTest(key=key):
                with self.assertRaises(ManifestError):
                    load_manifest_bytes(canonical_bytes(changed))
        changed = copy.deepcopy(self.value)
        changed["resolved"]["model.head-size"] += 1
        with self.assertRaises(ManifestError):
            load_manifest_bytes(canonical_bytes(changed))
        changed = copy.deepcopy(self.value)
        changed["resolved"]["model.kv-head-count"] = 3
        with self.assertRaises(ManifestError):
            load_manifest_bytes(canonical_bytes(changed))


if __name__ == "__main__":
    unittest.main()
