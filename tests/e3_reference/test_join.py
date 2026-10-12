from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from tests.e3_metrics.reference_probe import accumulate, divide, finalize, reduce_bool
from tests.e3_reference.corpus import PROFILE, materialize
from tests.e3_reference.join import ROLE_NAMES, ROOT, validate_five_case_join
from tests.e3_reference.transcript import TranscriptError, decode, f32_bits, f64_bits


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case(source: dict[str, object]) -> dict[str, object]:
    mathematical = []
    observed = []
    state = (0, 0, 0, 0, 0)
    zero32 = f32_bits(0.0)
    zero64 = f64_bits(0.0)
    ce32 = f32_bits(math.log(256))
    ce64 = f64_bits(math.log(256))
    for index, row in enumerate(source["batches"]):
        mask = row["mask"]
        targets = row["targets"]
        ce_bits = (int(ce32, 16), int(ce32, 16))
        numerator, weight = reduce_bool(ce_bits, tuple(mask))
        correct_count = sum(mask[position] and targets[position] == 0 for position in range(2))
        correct = int(f32_bits(float(correct_count)), 16)
        state = accumulate(state, (numerator, weight, correct, sum(mask)))
        observed.append({
            "ce_f32": [ce32, ce32],
            "correct_f32": f"{correct:08x}",
            "counters": list(state[3:]),
            "inputs": list(row["inputs"]),
            "logits_f32": [zero32] * 512,
            "mask": list(mask),
            "ordinal": index,
            "reduction_f32": [f"{word:08x}" for word in (
                numerator, weight, divide(numerator, weight)
            )],
            "state_f32": [f"{word:08x}" for word in state[:3]],
            "targets": list(targets),
        })
        roles = []
        for name in ROLE_NAMES:
            shape = ([1, 2, 256] if name == "Z" else
                     [1, 2, 8] if name in {"FU", "FG"} else
                     [1, 2, 2, 2] if name in {"QH", "KH", "VH", "AH"} else
                     [1, 2, 4])
            roles.append({"name": name, "shape": shape,
                          "values_f64": [zero64] * math.prod(shape)})
        mathematical.append({
            "ce_f64": [ce64, ce64], "inputs": list(row["inputs"]),
            "mask": list(mask), "ordinal": index, "roles": roles,
            "targets": list(targets),
        })
    return {
        "corpus": copy.deepcopy(source),
        "mathematical": mathematical,
        "mathematical_loss_f64": ce64,
        "name": source["name"],
        "synthetic": {
            "batches": observed,
            "final": {
                "counters": list(state[3:]),
                "metrics_f32": [f"{word:08x}" for word in finalize(state)],
            },
            "kind": "synthetic-observation",
            "profile": PROFILE,
        },
    }


def _bundle(corpus: dict[str, object]) -> dict[str, object]:
    sources = {
        "corpus_source_sha256": ROOT / "tests/e3_reference/corpus.py",
        "generator_source_sha256": ROOT / "tests/e3_reference/reference.py",
        "initializer_source_sha256": ROOT / "tests/n3k/test_initializer_reference.py",
        "m3_reference_source_sha256": ROOT / "tests/m3/reference.py",
        "q0_lock_sha256": ROOT / "tests/q0/requirements-oracle.lock",
        "rational_reference_source_sha256": ROOT / "tests/e3_metrics/reference_probe.py",
    }
    return {
        "cases": [_case(source) for source in corpus["cases"]],
        "kind": "reference-bundle",
        "model": {
            "initializer_seed": 1729,
            "initializer_successor": [1, 1729, 290, 0],
            "parameter_f32_sha256": "0" * 64,
            "reference_manifest_sha256": _sha(ROOT / "tests/m3/reference_manifest.json"),
        },
        "profile": json.loads((ROOT / "tests/m3/reference_manifest.json").read_text())["profile"],
        "provenance": {**{key: _sha(path) for key, path in sources.items()},
                       "python": "3.14.6", "torch": "2.13.0+cpu"},
    }


class FiveCaseJoinTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="e3-five-join-")
        self.root = Path(self.temporary.name) / "corpus"
        self.corpus = materialize(self.root)
        self.bundle = _bundle(self.corpus)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def check_failure(self, mutation, message: str) -> None:
        corpus = copy.deepcopy(self.corpus)
        bundle = copy.deepcopy(self.bundle)
        mutation(corpus, bundle)
        with self.assertRaisesRegex((TranscriptError, ValueError), message):
            validate_five_case_join(self.root, corpus, bundle)

    def test_five_physical_cases_join_and_return_exact_endpoints(self) -> None:
        joined = validate_five_case_join(self.root, self.corpus, self.bundle)
        self.assertEqual([item["name"] for item in joined],
                         ["packed-single", "packed-sharded", "packed-suffix",
                          "unpacked-three", "unpacked-pairs"])
        self.assertEqual(joined[0]["rows"], joined[1]["rows"])
        self.assertEqual(joined[2]["rows"], joined[0]["rows"][1:])
        self.assertEqual(joined[1]["end_cursor"], joined[2]["end_cursor"])
        self.assertNotEqual(joined[0]["end_cursor"], joined[1]["end_cursor"])
        self.assertEqual([sum(row["mask"]) for row in joined[4]["rows"]], [1, 1, 1])

    def test_case_and_schema_mutations_fail(self) -> None:
        self.check_failure(lambda c, b: b["cases"].pop(), "case count")
        self.check_failure(lambda c, b: b["cases"][1].update(name="packed-single"),
                           "case ID")
        self.check_failure(lambda c, b: b["cases"][0].update(extra=1), "case ID")
        self.check_failure(lambda c, b: b["cases"][0]["mathematical"][0]
                           ["roles"][4].update(name="ET"), "role name")
        self.check_failure(lambda c, b: b["cases"][0]["mathematical"][0]
                           ["roles"][4]["shape"].append(1), "shape")
        self.check_failure(lambda c, b: b["cases"][0]["mathematical"][0]
                           ["roles"][4]["values_f64"].pop(), "shape")
        self.check_failure(lambda c, b: b["cases"][1]["mathematical"][0]
                           ["roles"][0]["values_f64"].__setitem__(0, f64_bits(1.0)),
                           "packed repartition")

    def test_rows_cursors_digests_and_metrics_fail(self) -> None:
        self.check_failure(lambda c, b: b["cases"][0]["corpus"].update(
            start_cursor=c["cases"][2]["start_cursor"]), "corpus join")
        self.check_failure(lambda c, b: b["cases"][0]["synthetic"]
                           ["batches"][0]["targets"].reverse(), "targets")
        self.check_failure(lambda c, b: b["cases"][0]["synthetic"]
                           ["batches"][0]["ce_f32"].pop(), "ce_f32")
        self.check_failure(lambda c, b: b["cases"][0]["synthetic"]
                           ["batches"][1]["state_f32"].reverse(), "state_f32")
        self.check_failure(lambda c, b: b["cases"][0]["synthetic"]
                           ["final"]["metrics_f32"].pop(), "four words")
        self.check_failure(lambda c, b: c["cases"][0]["corpus"]
                           ["files"][0].update(sha256="0" * 64), "corpus join")
        self.check_failure(lambda c, b: b["model"].update(
            reference_manifest_sha256="0" * 64), "manifest digest")

    def test_physical_byte_change_and_duplicate_json_key_fail(self) -> None:
        path = self.root / "packed-single" / "shard-0000000000000000.ets"
        raw = bytearray(path.read_bytes())
        raw[90] ^= 1
        path.write_bytes(raw)
        with self.assertRaisesRegex(TranscriptError, "physical D1"):
            validate_five_case_join(self.root, self.corpus, self.bundle)
        with self.assertRaisesRegex(TranscriptError, "duplicate JSON key"):
            decode(b'{"format":"eshkol-e3-development-transcript",'
                   b'"format":"eshkol-e3-development-transcript"}\n')


if __name__ == "__main__":
    unittest.main()
