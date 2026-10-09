"""Source, arithmetic, and near-miss controls for independent C4 math."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import torch

from tests.g3c4 import c4_math_oracle as oracle


def synthetic_weights() -> tuple[torch.Tensor, ...]:
    weights = [torch.zeros(shape, dtype=torch.float64) for shape in oracle.SHAPES]
    weights[10] = ((torch.arange(1024, dtype=torch.float64).reshape(256, 4) % 23)
                   - 11) / 9
    weights[13] = torch.tensor([
        [.3, -.2, .1, .4], [-.5, .2, .3, -.1],
        [.2, .4, -.3, .1], [-.1, .3, .2, -.4]], dtype=torch.float64)
    for index in (7, 9, 12):
        weights[index] = torch.tensor([1., .8, 1.2, .7], dtype=torch.float64)
    for index in (6, 8, 11):
        weights[index] = torch.tensor([.1, -.1, .05, .2], dtype=torch.float64)
    for index in (0, 1, 2, 3):
        weights[index] = torch.tensor([
            [.4, -.1, .2, .3], [-.2, .3, .5, -.1],
            [.1, .7, -.3, .2], [.5, -.4, .2, .6]],
            dtype=torch.float64) * (index + 1) / 3
    weights[5] = ((torch.arange(32, dtype=torch.float64).reshape(8, 4) % 7)
                  - 3) / 4
    weights[4] = ((torch.arange(32, dtype=torch.float64).reshape(4, 8) % 9)
                  - 4) / 5
    return tuple(weights)


def observed_from_expected(fixture: dict) -> dict:
    observed = {"version": 1, "models": copy.deepcopy(fixture["models"]),
                "cases": []}
    for case in fixture["cases"]:
        row = {"seed": case["seed"], "tokens": case["tokens"]}
        for name in ("logits", "keys", "values"):
            record = case[name]
            row[name] = {"dtype": "float32", "device": "cpu",
                         "shape": record["shape"],
                         "bits": [struct.pack(">f", struct.unpack(
                             ">d", bytes.fromhex(word))[0]).hex()
                                  for word in record["bits"]]}
        observed["cases"].append(row)
    return observed


class C4MathOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        oracle.require_q0()

    def test_pinned_source_and_all_real_drift_channels(self) -> None:
        oracle.check_source_pins()
        for path in ("native/g3c4_primitives_provider.c",
                     "src/eshkol_transformer/g3c4_model_owner.c",
                     "tests/g3s/reference.py"):
            with self.subTest(path=path):
                changed = (oracle.ROOT / path).read_bytes() + b"\n"
                with self.assertRaisesRegex(ValueError, "drift"):
                    oracle.check_source_pins(overrides={path: changed})
        manifest = "native/g3c4_full_prefix_forward_source_closure.txt"
        rows = (oracle.ROOT / manifest).read_text().splitlines()
        for changed in (rows[1:2] + rows[:1] + rows[2:], rows[:-1]):
            with self.assertRaisesRegex(ValueError, "drift"):
                oracle.check_source_pins(overrides={
                    manifest: ("\n".join(changed) + "\n").encode()})
        with self.assertRaisesRegex(ValueError, "pin manifest drift"):
            oracle.check_source_pins(source_pins=(
                oracle.ROOT / "tests/g3c4/c4_math_source_pins.sha256"
            ).read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "original ROADMAP"):
            oracle.check_source_pins(roadmap_basis=b"wrong basis")
        with self.assertRaisesRegex(ValueError, "successor ROADMAP"):
            oracle.check_source_pins(overrides={
                "docs/ROADMAP.md": (
                    oracle.ROOT / "docs/ROADMAP.md").read_bytes() + b"\n"})
        with self.assertRaisesRegex(ValueError, "successor ROADMAP diff"):
            oracle.check_source_pins(roadmap_delta=b"wrong diff")

    def test_philox_initializer_shape_tie_and_counter(self) -> None:
        self.assertEqual(oracle.philox(0, 0),
                         (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8))
        self.assertEqual(oracle.weight_bits(0), 0xBD000000)
        self.assertEqual(oracle.weight_bits(0x80000000), 0)
        self.assertEqual(oracle.PATHS[10], "head/weight")
        self.assertNotIn("token_embedding/weight", oracle.PATHS)
        for seed in oracle.SEEDS:
            bits, successor = oracle.initialize(seed)
            self.assertEqual(sum(map(len, bits)), 1192)
            self.assertEqual(successor, (1, seed, 292, 0))
            self.assertEqual(tuple(oracle.tensors(bits)[13].shape), (4, 4))
            for index in (6, 8, 11):
                self.assertEqual(bits[index], [0] * 4)
            for index in (7, 9, 12):
                self.assertEqual(bits[index], [0x3F800000] * 4)
        with self.assertRaises(ValueError):
            oracle.initialize(-1)
        changed = copy.deepcopy(oracle.initialize(1729)[0])
        changed[10][0] ^= 1
        self.assertNotEqual(changed, oracle.initialize(1729)[0])
        wrong_order = list(oracle.PATHS)
        wrong_order[0], wrong_order[1] = wrong_order[1], wrong_order[0]
        with self.assertRaisesRegex(ValueError, "schedule"):
            oracle.check_schedule(tuple(wrong_order), oracle.SHAPES,
                                  "head/weight")
        with self.assertRaisesRegex(ValueError, "schedule"):
            oracle.check_schedule(oracle.PATHS, oracle.SHAPES,
                                  "token_embedding/weight")

    def test_all_prefix_lengths_shapes_and_bad_inputs(self) -> None:
        weights = oracle.tensors(oracle.initialize(1729)[0])
        for tokens in ((0,), (255, 255), (0, 255, 7), (7, 11, 0, 255)):
            logits, keys, values = oracle.forward(weights, tokens)
            self.assertEqual(tuple(logits.shape), (len(tokens), 256))
            self.assertEqual(tuple(keys.shape), (1, 2, len(tokens), 2))
            self.assertEqual(tuple(values.shape), (1, 2, len(tokens), 2))
        for tokens in ((), (0, 1, 2, 3, 4), (-1,), (256,), (True,)):
            with self.assertRaises(ValueError):
                oracle.forward(weights, tokens)
        wrong = list(weights)
        wrong[13] = wrong[13][:3]
        with self.assertRaises(ValueError):
            oracle.forward(tuple(wrong), (0, 255))
        with self.assertRaisesRegex(ValueError, "variant"):
            oracle.forward(weights, (0, 255), "unreviewed")
        wrong = list(weights)
        wrong[10] = wrong[10].to(torch.float32)
        with self.assertRaises(ValueError):
            oracle.forward(tuple(wrong), (0, 255))

    def test_actual_equation_mutants_separate(self) -> None:
        weights = synthetic_weights()
        tokens = (0, 255, 7, 11)
        accepted = oracle.forward(weights, tokens)
        for mutation in (
            "no_position", "wrong_norm_variance", "wrong_norm_swap",
            "wrong_matrix_orientation", "wrong_qk", "wrong_head_split",
            "wrong_scale", "wrong_causal", "wrong_gelu", "wrong_head_tie",
        ):
            with self.subTest(mutation=mutation):
                mutated = oracle.forward(weights, tokens, mutation)
                maximum = max(float((before - after).abs().max())
                              for before, after in zip(accepted, mutated))
                self.assertGreater(maximum, 2e-4 if mutation == "wrong_gelu"
                                   else 1e-4)

    def test_two_fresh_processes_and_wrong_transcripts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first, second = (Path(directory) / name for name in
                             ("first.json", "second.json"))
            command = [sys.executable, "-m", "tests.g3c4.c4_math_oracle",
                       "--output"]
            for output in (first, second):
                subprocess.run(command + [str(output)], check=True,
                               capture_output=True, timeout=60)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            fixture = json.loads(first.read_bytes())
            frozen_sha256 = hashlib.sha256(first.read_bytes()).hexdigest()
            self.assertEqual(len(fixture["terminal_diagnostics"]), 4)
            self.assertEqual({len(case["tokens"]) for case in fixture["cases"]},
                             {1, 2, 3, 4})
            observed = observed_from_expected(fixture)
            self.assertLess(oracle.compare_observation(
                fixture, observed, frozen_sha256)["maximum_absolute_error"],
                1e-6)
            observation = Path(directory) / "observed.json"
            observation.write_bytes(oracle.encode(observed))
            comparison = subprocess.run(
                [sys.executable, "-m", "tests.g3c4.c4_math_oracle",
                 "--fixture", str(first), "--frozen-sha256", frozen_sha256,
                 "--observation", str(observation)], check=True,
                capture_output=True, timeout=60)
            self.assertIn(b"maximum_absolute_error", comparison.stdout)
            duplicate = subprocess.run(command + [str(first)], check=False,
                                       capture_output=True, timeout=60)
            self.assertNotEqual(duplicate.returncode, 0)
            self.assertIn(b"overwrite", duplicate.stderr)
            for field in ("logits", "keys", "values"):
                changed = copy.deepcopy(observed)
                changed["cases"][0][field]["bits"][0] = "3f800000"
                with self.assertRaisesRegex(ValueError, "numerical drift"):
                    oracle.compare_observation(fixture, changed, frozen_sha256)
            for field in ("dtype", "device", "shape"):
                changed = copy.deepcopy(observed)
                changed["cases"][0]["logits"][field] = (
                    "float64" if field == "dtype" else
                    "cuda" if field == "device" else [2, 256])
                with self.assertRaises(ValueError):
                    oracle.compare_observation(fixture, changed, frozen_sha256)
            changed = copy.deepcopy(observed)
            changed["models"][0]["parameters"]["bits"][0] = "3f800000"
            with self.assertRaisesRegex(ValueError, "initializer"):
                oracle.compare_observation(fixture, changed, frozen_sha256)
            changed = copy.deepcopy(observed)
            changed["models"][0]["successor_words"][2] = 293
            with self.assertRaisesRegex(ValueError, "initializer"):
                oracle.compare_observation(fixture, changed, frozen_sha256)
            changed = copy.deepcopy(observed)
            changed["cases"].pop()
            with self.assertRaisesRegex(ValueError, "count"):
                oracle.compare_observation(fixture, changed, frozen_sha256)
            changed = copy.deepcopy(fixture)
            changed["absolute_tolerance"] = 1.0
            with self.assertRaisesRegex(ValueError, "frozen fixture"):
                oracle.compare_observation(changed, observed, frozen_sha256)
            for raw in (b'{"a":1,"a":2}\n', b'{"x":NaN}\n',
                        b'{ "x":1 }\n', b"x" * (16 * 1024 * 1024 + 1)):
                with self.assertRaises(ValueError):
                    oracle.decode(raw)


if __name__ == "__main__":
    unittest.main()
