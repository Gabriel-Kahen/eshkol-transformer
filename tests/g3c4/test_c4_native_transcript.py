"""Synthetic transport negatives; no native binary or observed math is run."""

from __future__ import annotations

import copy
import hashlib
import tempfile
from pathlib import Path
import unittest
from unittest import mock

from tests.g3c4 import c4_native_transcript as capture


def fixture() -> dict:
    prompts = ([0], [255], [0, 255], [7, 11], [0, 255, 7],
               [7, 11, 0], [0, 255, 7, 11], [7, 11, 0, 255])
    rows = {0: prompts + prompts[:2],
            1729: prompts + prompts + prompts[:2],
            capture.SEEDS[2]: prompts + prompts[:2]}
    return {"models": [{"seed": seed} for seed in capture.SEEDS],
            "cases": [{"seed": seed, "tokens": list(tokens)}
                      for seed in capture.SEEDS for tokens in rows[seed]]}


def raw(frozen: dict) -> bytes:
    lines = ["C4CAP1"]
    for model in frozen["models"]:
        seed = model["seed"]
        lines.extend((f"MODEL {seed} 1 {seed} 292 0",
                      "PARAMETERS 1192 " + " ".join(["00000000"] * 1192)))
        for row in (case for case in frozen["cases"] if case["seed"] == seed):
            tokens = row["tokens"]
            p = len(tokens)
            lines.append(f"CASE {seed} {p} " + " ".join(map(str, tokens)))
            if p == 4:
                lines.append("T4ABORT 1 0")
            for tag, count in (("LOGITS", p * 256), ("KEYS", p * 4),
                               ("VALUES", p * 4)):
                lines.append(f"{tag} {count} " + " ".join(["00000000"] * count))
            lines.append("ENDCASE")
    lines.append("ENDCAPTURE")
    return ("\n".join(lines) + "\n").encode("ascii")


class NativeTranscriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frozen = fixture()
        self.raw = raw(self.frozen)

    def test_exact_transport_schema_and_request(self) -> None:
        request = capture.make_input(self.frozen)
        self.assertEqual(request.count(b"MODEL"), 3)
        self.assertEqual(request.count(b"CASE"), len(self.frozen["cases"]))
        observed = capture.parse(self.raw, self.frozen)
        self.assertEqual([model["successor_words"] for model in observed["models"]],
                         [[1, seed, 292, 0] for seed in capture.SEEDS])
        self.assertEqual(observed["cases"][7]["values"]["shape"], [1, 2, 4, 2])
        self.assertEqual(capture.encode_observation(observed)[-1:], b"\n")

    def test_frozen_hash_and_canonical_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            payload = capture.encode_observation(self.frozen)
            path.write_bytes(payload)
            with mock.patch.object(capture, "FIXTURE_SHA256",
                                   hashlib.sha256(payload).hexdigest()):
                self.assertEqual(capture.load_fixture(path), self.frozen)
                path.write_bytes(payload + b" ")
                with self.assertRaisesRegex(ValueError, "SHA drift"):
                    capture.load_fixture(path)

    def test_identity_order_count_and_t4_abort_negatives(self) -> None:
        bad = (
            self.raw.replace(b"MODEL 1729", b"MODEL 1730", 1),
            self.raw.replace(b"CASE 1729 2 0 255", b"CASE 1729 2 255 0", 1),
            self.raw.replace(b"T4ABORT 1 0", b"T4ABORT 0 0", 1),
            self.raw.replace(b"ENDCASE\n", b"ENDCASE\nENDCASE\n", 1),
            self.raw.replace(b"LOGITS 256", b"LOGITS 255", 1),
            self.raw[:-1],
            self.raw + b"EXTRA\n",
        )
        for damaged in bad:
            with self.subTest(digest=hashlib.sha256(damaged).hexdigest()[:8]):
                with self.assertRaises(ValueError):
                    capture.parse(damaged, self.frozen)

    def test_bits_and_state_negatives(self) -> None:
        for old, new in ((b"PARAMETERS 1192 00000000", b"PARAMETERS 1192 7f800000"),
                         (b"KEYS 4 00000000", b"KEYS 4 7fc00000"),
                         (b"MODEL 0 1 0 292 0", b"MODEL 0 1 0 291 0"),
                         (b"VALUES 4 00000000", b"VALUES 4 0000000G")):
            with self.subTest(replacement=new[:30]):
                self.assertIn(old, self.raw)
                with self.assertRaises(ValueError):
                    capture.parse(self.raw.replace(old, new, 1), self.frozen)

    def test_request_rejects_noninteger_prefix(self) -> None:
        changed = copy.deepcopy(self.frozen)
        changed["cases"][0]["tokens"][0] = False
        with self.assertRaisesRegex(ValueError, "prefix identity"):
            capture.make_input(changed)

    def test_real_child_exit_timeout_and_drift_retain_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = {
                "exit": "#!/bin/sh\nprintf failure >&2\nexit 7\n",
                "timeout": "#!/bin/sh\nsleep 2\n",
                "drift": "#!/bin/sh\nprintf x >> \"$0\"\nprintf C4CAP1\\n\n",
            }
            for name, source in scripts.items():
                with self.subTest(name=name):
                    binary = root / name
                    binary.write_text(source)
                    binary.chmod(0o700)
                    receipt = root / f"{name}-receipt"
                    with self.assertRaises(ValueError):
                        capture.run_native(binary, b"", receipt, timeout_seconds=1)
                    self.assertTrue(receipt.with_suffix(".json").is_file())
                    self.assertTrue(receipt.with_suffix(".stdout").is_file())
                    self.assertTrue(receipt.with_suffix(".stderr").is_file())


if __name__ == "__main__":
    unittest.main()
