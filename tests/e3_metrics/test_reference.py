"""Real-provider transcript versus frozen PyTorch and independent rational math."""
from __future__ import annotations
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tests.q0.oracle_format import decode_tensor, load_fixture, tensor_by_name
from reference_probe import (EXP_NEIGHBORS, INF_BITS, accumulate, divide, exp_bits,
                             finalize, reduce_bool, rounded, value)


def run(runner, *args):
    result = subprocess.run([str(runner), *args], capture_output=True, text=True, timeout=60)
    if result.returncode or result.stderr:
        raise AssertionError(f"native runner failed: {result.returncode}: {result.stderr}")
    fields = {}
    for line in result.stdout.splitlines():
        name, *values = line.split()
        if name in fields: raise AssertionError("duplicate native field")
        fields[name] = tuple(int(x, 10 if name.endswith(".counters") or name == "status" else 16) for x in values)
    return fields


def f32(bits):
    return struct.unpack(">f", bits.to_bytes(4, "big"))[0]


def check_transcript(payload, runner):
    actual = run(runner, "--emit")
    again = run(runner, "--emit")
    if actual != again: raise AssertionError("nondeterministic numerical transcript")
    expected_fields = {f"batch{b}.{field}" for b in range(3) for field in ("ce", "reduction", "correct", "state", "counters")} | {"global.metrics"}
    if set(actual) != expected_fields: raise AssertionError("unexpected transcript schema")
    state = (0, 0, 0, 0, 0)
    for b in range(3):
        prefix = f"batch{b}"
        losses = actual[prefix + ".ce"]
        mask = decode_tensor(tensor_by_name(payload, prefix + ".mask"))
        frozen_ce = decode_tensor(tensor_by_name(payload, prefix + ".ce"))
        mathematical_ce = decode_tensor(tensor_by_name(payload, prefix + ".mathematical_ce"))
        for i, word in enumerate(losses):
            for expected in (frozen_ce[i], mathematical_ce[i]):
                if abs(f32(word) - expected) > 2e-6 + 2e-5 * abs(expected):
                    raise AssertionError(f"CE parity {prefix}[{i}]")
        bn, bw = reduce_bool(losses, mask)
        if actual[prefix + ".reduction"] != (bn, bw, divide(bn, bw)):
            raise AssertionError("rational L3S numerator/weight/mean mismatch")
        correct = tuple(int(x, 16) for x in tensor_by_name(payload, prefix + ".correct")["data"])
        if actual[prefix + ".correct"] != correct: raise AssertionError("numeric correctness mismatch")
        state = accumulate(state, (bn, bw, correct[0], sum(mask)))
        if actual[prefix + ".state"] != state[:3]: raise AssertionError("numeric accumulation order/value mismatch")
        if actual[prefix + ".counters"] != state[3:]: raise AssertionError("numeric counter mismatch")
    expected = finalize(state); metrics = actual["global.metrics"]
    if (metrics[0], metrics[1], metrics[3]) != (expected[0], expected[1], expected[3]):
        raise AssertionError("numeric global finalization mismatch")
    if abs(metrics[2] - expected[2]) > 2: raise AssertionError("Decimal exp exceeds two f32 ULP")
    math_loss = decode_tensor(tensor_by_name(payload, "global.mathematical_loss"))[0]
    if abs(f32(metrics[0]) - math_loss) > 2e-6 + 2e-5 * abs(math_loss):
        raise AssertionError("separate mathematical global loss mismatch")
    return actual


class Reference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = load_fixture(FIXTURE)

    def test_source_pins_and_checksums(self):
        digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        g = self.payload["generator"]
        self.assertEqual(g["source_sha256"], digest(Path(__file__).with_name("generate_reference.py")))
        self.assertEqual(g["dependency_lock_sha256"], digest(ROOT / "tests/q0/requirements-oracle.lock"))
        self.assertEqual(g["framework"], {"name": "pytorch", "version": "2.13.0+cpu"})
        provenance = json.loads(FIXTURE.with_suffix(".provenance.json").read_text())
        self.assertEqual(provenance, dict(schema="e3-metrics-oracle-provenance-v1", python="3.14.6", torch="2.13.0+cpu",
                                         eshkol_source_commit="81298b4a9608fb92eb6f351a2eabd8392da7d9ef",
                                         toolchain_lock_sha256=digest(ROOT / "toolchain/eshkol.lock"),
                                         generator_sha256=g["source_sha256"], fixture_sha256=digest(FIXTURE)))
        self.assertEqual([decode_tensor(tensor_by_name(self.payload, f"batch{b}.mask")) for b in range(3)],
                         [(True, True), (False, True), (True, False)])

    def test_real_carrier_transcript_and_rational_rounding(self):
        check_transcript(self.payload, RUNNER)

    def test_frozen_exp_neighbors_and_sampled_ulps(self):
        maximum_ulp = 0
        inputs = list(EXP_NEIGHBORS) + [0, 1, 0x00800000, rounded(Fraction(1, 3)), rounded(1), rounded(2), rounded(20), rounded(64), rounded(88)]
        inputs += [rounded(Fraction(i, 7)) for i in range(1, 620, 7)]
        for n in inputs:
            expected = exp_bits(n)
            result = run(RUNNER, "--finalize", f"{n:08x}", "3f800000", "00000000")
            if expected == INF_BITS:
                self.assertNotEqual(result["status"][0], 0)
                self.assertEqual(result["final"], (0x4ABC1234,) * 4)
            else:
                self.assertEqual(result["status"], (0, 0, 0))
                self.assertEqual(result["final"][0], n)
                self.assertEqual(result["final"][1], rounded(1))
                self.assertEqual(result["final"][3], 0)
                ulp = abs(result["final"][2] - expected)
                maximum_ulp = max(maximum_ulp, ulp)
                self.assertLessEqual(ulp, 2)
        print(f"E3 Decimal.exp oracle: {len(inputs)} exact-f32 inputs, maximum {maximum_ulp} ULP; frozen overflow neighbors pass")

    def test_zero_subnormal_underflow_and_cap(self):
        for n, w, c in ((0x80000000, rounded(1), 0x80000000), (1, rounded(16384), 0),
                        (1, rounded(1), 0), (rounded(1), rounded(16384), rounded(1))):
            actual = run(RUNNER, "--finalize", f"{n:08x}", f"{w:08x}", f"{c:08x}")
            self.assertEqual(actual["status"], (0, 0, 0))
            expected = finalize((n, w, c))
            self.assertEqual(actual["final"][0:2], expected[0:2])
            self.assertEqual(actual["final"][3], expected[3])
            self.assertLessEqual(abs(actual["final"][2] - expected[2]), 2)

    def test_physical_three_batch_witness(self):
        actual = run(RUNNER, "--witness")
        self.assertEqual(actual["witness.state"], (0x42800000, rounded(3), 0))
        self.assertEqual(actual["witness.final"][0], 0x41AAAAAB)
        self.assertLessEqual(abs(actual["witness.final"][2] - exp_bits(0x41AAAAAB)), 2)
        means = run(RUNNER, "--mean-witness")
        self.assertEqual(means["witness.state"], (rounded(7), rounded(3), 0))
        self.assertEqual(means["witness.final"][0], rounded(Fraction(7, 3)))


if __name__ == "__main__":
    FIXTURE, RUNNER = map(Path, sys.argv[1:3])
    del sys.argv[1:3]
    unittest.main()
