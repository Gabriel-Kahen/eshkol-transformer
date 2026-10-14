"""Source-only checks for the frozen TR3 successor contract witness."""

from __future__ import annotations

import hashlib
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "tests/tr3_lease/test_restore_successor_contract.py"
MANIFESTS = (
    ROOT / "tests/tr3_lease/runtime_candidate.tsv",
    ROOT / "tests/tr3_lease/failure_runtime_candidate.tsv",
)
KEY = "successor_contract_sha256"
STALE_PIN = "10f8d06e5beedaee0ba113831d290196f24683c76192721a055640f5b13f01e1"


def pinned_digest(manifest: str) -> str:
    entries = [line.split("\t", 1) for line in manifest.splitlines()]
    if any(len(entry) != 2 for entry in entries):
        raise ValueError("malformed TR3 manifest")
    pins = [value for key, value in entries if key == KEY]
    if len(pins) != 1 or re.fullmatch(r"[0-9a-f]{64}", pins[0]) is None:
        raise ValueError("missing or malformed TR3 successor pin")
    return pins[0]


class RuntimeManifestPins(unittest.TestCase):
    def test_both_manifests_pin_the_unchanged_successor_contract(self) -> None:
        contract_digest = hashlib.sha256(CONTRACT.read_bytes()).hexdigest()
        for path in MANIFESTS:
            with self.subTest(manifest=path.name):
                self.assertEqual(pinned_digest(path.read_text()), contract_digest)

    def test_stale_and_foreign_pins_are_rejected(self) -> None:
        contract_digest = hashlib.sha256(CONTRACT.read_bytes()).hexdigest()
        foreign_pin = hashlib.sha256(b"foreign TR3 successor contract").hexdigest()
        self.assertNotIn(contract_digest, (STALE_PIN, foreign_pin))
        for path in MANIFESTS:
            original = path.read_text()
            current_pin = pinned_digest(original)
            for bad_pin in (STALE_PIN, foreign_pin):
                with self.subTest(manifest=path.name, pin=bad_pin):
                    altered = original.replace(
                        f"{KEY}\t{current_pin}", f"{KEY}\t{bad_pin}", 1
                    )
                    self.assertNotEqual(pinned_digest(altered), contract_digest)


if __name__ == "__main__":
    unittest.main()
