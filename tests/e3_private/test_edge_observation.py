from __future__ import annotations

import copy
import hashlib
import math
import struct
import tempfile
import unittest
from pathlib import Path

from tests.e3_private.parse_edge_observation import parse
from tests.e3_private.parse_numerical_observation import ROLE_NAMES
from tests.e3_reference.edge_corpus import materialize
from tests.e3_reference.transcript import f64_bits
from tests.e3_metrics.reference_probe import accumulate, divide, finalize, reduce_bool


def failure_line(name: str, cursor: str, *, after: str = "0,0,0,0,0,0") -> bytes:
    return (f"E3EDGE\t{name}\tinvalid-state\t0,0,0,0,0,0\t{after}"
            f"\t{cursor}\t{cursor}\nE3-EDGE-RUNTIME-PASS\n").encode()


def _f32(value: float) -> int:
    return int.from_bytes(struct.pack(">f", value), "big")


def _word(value: int) -> str:
    return f"{value:08x}"


def success_control(root: Path, manifest: dict[str, object]) -> tuple[bytes, dict[str, object]]:
    """Uniform synthetic parser control, deliberately not an M3/Q0 proof."""
    case = manifest["cases"][3]
    ce = _f32(math.log(256))
    logits = ["00000000"] * 512
    lines: list[str] = []
    state = (0, 0, 0, 0, 0)
    mathematical = []
    for ordinal, row in enumerate(case["batches"]):
        pair = lambda values: ",".join(str(value) for value in values)
        lines.append(f"E3ROW\t{ordinal}\t{pair(row['inputs'])}\t"
                     f"{pair(row['targets'])}\t{pair(int(x) for x in row['mask'])}")
        for role in ROLE_NAMES:
            count = 512 if role == "Z" else 2
            shape = "1,2,256" if role == "Z" else "1,2,1"
            lines.append(f"E3ROLE\t{ordinal}\t{role}\t{shape}\t"
                         + ",".join(["00000000"] * count))
        lines.append(f"E3CE\t{ordinal}\t{_word(ce)},{_word(ce)}")
        numerator, weight = reduce_bool((ce, ce), tuple(row["mask"]))
        correct = _f32(sum(mask and target == 0 for mask, target in
                           zip(row["mask"], row["targets"])))
        reduction = (numerator, weight, divide(numerator, weight), correct)
        lines.append(f"E3BATCH\t{ordinal}\t"
                     + ",".join(map(_word, reduction)) + f"\t{sum(row['mask'])}")
        state = accumulate(state, (numerator, weight, correct, sum(row["mask"])))
        lines.append(f"E3STATE\t{ordinal}\t"
                     + ",".join(map(_word, state[:3]))
                     + f"\t{state[3]},{state[4]}")
        mathematical.append({"inputs": row["inputs"], "targets": row["targets"],
                             "mask": row["mask"],
                             "roles": [{"name": "Z", "values_f64": [f64_bits(0.0)] * 512}],
                             "ce_f64": [f64_bits(math.log(256))] * 2})
    final = finalize(state)
    lines.append("E3FINAL\t" + ",".join(map(_word, final))
                 + f"\t{state[3]},{state[4]}")
    after = ",".join(map(str, (*final, state[3], state[4])))
    cursor = case["start_cursor"]
    lines.append(f"E3EDGE\tone-valid-final\tsuccess\t0,0,0,0,0,0\t"
                 f"{after}\t{cursor}\t{cursor}")
    lines.append("E3-EDGE-RUNTIME-PASS")
    oracle = {
        "kind": "e3-edge-oracle-v1", "case": "one-valid-final", "cursor": cursor,
        "model_seed": 1729, "python": "3.14.6", "torch": "2.13.0+cpu",
        "corpus_sha256": hashlib.sha256((root / "edge-corpus.json").read_bytes()).hexdigest(),
        "m3_reference_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "m3/reference.py").read_bytes()).hexdigest(),
        "q0_lock_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "q0/requirements-oracle.lock").read_bytes()).hexdigest(),
        "mathematical": mathematical,
        "mathematical_loss_f64": f64_bits(math.log(256)),
    }
    return ("\n".join(lines) + "\n").encode(), oracle


class EdgeObservationTests(unittest.TestCase):
    def test_three_zero_weight_outcomes_and_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "corpus"
            manifest = materialize(root)
            for case in manifest["cases"][:3]:
                name, cursor = case["name"], case["start_cursor"]
                result = parse(failure_line(name, cursor), root, manifest, name)
                self.assertEqual(result["status"], "invalid-state")
                self.assertEqual(result["before"], result["after"])
                with self.subTest(name=name), self.assertRaises(ValueError):
                    parse(failure_line(name, cursor, after="0,0,0,0,1,0"),
                          root, manifest, name)
                with self.subTest(name=name), self.assertRaises(ValueError):
                    parse(failure_line(name, "00"), root, manifest, name)
                with self.subTest(name=name), self.assertRaises(ValueError):
                    parse(b"E3CE\t0\tbad\n" + failure_line(name, cursor),
                          root, manifest, name)

    def test_success_parser_reaches_mathematical_and_published_word_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "corpus"
            manifest = materialize(root)
            raw, oracle = success_control(root, manifest)
            result = parse(raw, root, manifest, "one-valid-final", oracle)
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["after"][-2:], [3, 2])
            with self.assertRaisesRegex(ValueError, "oracle"):
                parse(raw, root, manifest, "one-valid-final")
            with self.assertRaisesRegex(ValueError, "identity"):
                parse(raw, root, manifest, "one-valid-final", {})
            bad = copy.deepcopy(oracle)
            bad["cursor"] = "00"
            with self.assertRaisesRegex(ValueError, "identity"):
                parse(raw, root, manifest, "one-valid-final", bad)
            bad = copy.deepcopy(oracle)
            bad["mathematical"][1]["roles"][0]["values_f64"][0] = f64_bits(100.0)
            with self.assertRaisesRegex(ValueError, "mathematical tolerance"):
                parse(raw, root, manifest, "one-valid-final", bad)
            lines = raw.splitlines(keepends=True)
            fields = lines[-2].split(b"\t")
            words = fields[4].split(b",")
            words[4] = b"4"
            fields[4] = b",".join(words)
            lines[-2] = b"\t".join(fields)
            changed = b"".join(lines)
            with self.assertRaisesRegex(ValueError, "six published outputs"):
                parse(changed, root, manifest, "one-valid-final", oracle)


if __name__ == "__main__":
    unittest.main()
