from __future__ import annotations

import hashlib
import math
import re
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.q0 import generate_n2_primitives as generator
from tests.q0.oracle_format import decode_tensor, encode_fixture, load_fixture, tensor_by_name
from tests.n2 import generate_reference_header as header_generator

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
GENERATOR = PROJECT_ROOT / "tests" / "q0" / "generate_n2_primitives.py"
LOCK = HERE / "requirements-reference.lock"
FROZEN = PROJECT_ROOT / "tests" / "q0" / "fixtures" / "n2_primitives_v1.json"
FROZEN_SHA256 = "a32e065db6d7654600121696ba680cb15654c6712696e37fdb4a5cfb51abca03"
HEADER_GENERATOR = HERE / "generate_reference_header.py"
HEADER = HERE / "reference_vectors.h"

EXPECTED_CASE_OPERATIONS = {
    "dropout.eval.forward": "dropout.eval.forward",
    "dropout.eval.gradient": "dropout.eval.backward",
    "dropout.train.p0.forward": "dropout.train.forward",
    "dropout.train.p0.gradient": "dropout.train.backward",
    "dropout.train.p25.forward": "dropout.train.forward",
    "dropout.train.p25.gradient": "dropout.train.backward",
    "embedding.repeated.forward": "embedding.forward",
    "embedding.repeated.gradient": "embedding.backward",
    "gelu.exact.forward": "gelu.forward",
    "gelu.exact.gradient": "gelu.backward",
    "layer_norm.affine.forward": "layer-norm.forward",
    "layer_norm.affine.gradient": "layer-norm.backward",
    "layer_norm.d1.forward": "layer-norm.forward",
    "layer_norm.d1.gradient": "layer-norm.backward",
    "layer_norm.near_constant.forward": "layer-norm.forward",
    "layer_norm.near_constant.gradient": "layer-norm.backward",
    "linear.bias.forward": "linear.forward",
    "linear.bias.gradient": "linear.backward",
    "linear.no_bias.forward": "linear.forward-no-bias",
    "linear.no_bias.gradient": "linear.backward-no-bias",
    "relu.zero.forward": "relu.forward",
    "relu.zero.gradient": "relu.backward",
    "residual.distinct.forward": "residual.forward",
    "residual.distinct.gradient": "residual.backward",
    "residual.repeated.forward": "residual.forward",
    "residual.repeated.gradient": "residual.backward",
}


def _pinned_torch_available() -> bool:
    return generator.torch is not None and (
        generator.torch.__version__ == generator.PINNED_TORCH_VERSION
    )


class N2FrozenFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = FROZEN.read_bytes()
        cls.payload = load_fixture(cls.raw)

    def test_frozen_bytes_hash_and_complete_case_surface(self) -> None:
        self.assertEqual(hashlib.sha256(self.raw).hexdigest(), FROZEN_SHA256)
        self.assertEqual(len(self.payload["cases"]), 26)
        self.assertEqual(len(self.payload["tensors"]), 82)
        actual = {case["name"]: case["operation"] for case in self.payload["cases"]}
        self.assertEqual(actual, EXPECTED_CASE_OPERATIONS)

    def test_frozen_generator_and_lock_identities_match_sources(self) -> None:
        metadata = self.payload["generator"]
        self.assertEqual(metadata["source_sha256"],
                         hashlib.sha256(GENERATOR.read_bytes()).hexdigest())
        self.assertEqual(metadata["dependency_lock_sha256"],
                         hashlib.sha256(LOCK.read_bytes()).hexdigest())


class N2ReferenceHeaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = FROZEN.read_bytes()
        cls.payload = load_fixture(cls.fixture)
        cls.header = HEADER.read_bytes()

    def test_header_regeneration_is_byte_identical(self) -> None:
        self.assertEqual(header_generator.render_header(self.fixture), self.header)
        with tempfile.TemporaryDirectory() as temporary:
            outputs = [Path(temporary) / "first.h", Path(temporary) / "second.h"]
            for output in outputs:
                subprocess.run(
                    [sys.executable, str(HEADER_GENERATOR),
                     "--fixture", str(FROZEN), "--output", str(output)],
                    cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
                )
            self.assertEqual(outputs[0].read_bytes(), outputs[1].read_bytes())
            self.assertEqual(outputs[0].read_bytes(), self.header)

    def test_header_fixture_hash_and_complete_case_counts(self) -> None:
        source = self.header.decode("ascii")
        self.assertIn(hashlib.sha256(self.fixture).hexdigest(), source)
        self.assertIn("ET_N2_REFERENCE_TENSOR_COUNT UINT32_C(82)", source)
        self.assertIn("ET_N2_REFERENCE_CASE_COUNT UINT32_C(26)", source)
        for case in self.payload["cases"]:
            self.assertIn(f'"{case["name"]}", "{case["operation"]}"', source)

    def test_every_referenced_and_expectation_tensor_has_exact_words(self) -> None:
        source = self.header.decode("ascii")
        referenced = {
            name for case in self.payload["cases"]
            for name in case["inputs"] + case["expectation"]["outputs"]
        }
        expectations = {
            name for case in self.payload["cases"]
            for name in case["expectation"]["outputs"]
        }
        records = {tensor["name"]: tensor for tensor in self.payload["tensors"]}
        self.assertEqual(referenced, set(records))
        for name in sorted(referenced):
            tensor = records[name]
            symbol = header_generator.tensor_symbol(name)
            element_type = "uint32_t" if tensor["dtype"] == "float32" else "int64_t"
            match = re.search(
                rf"static const {element_type} {symbol}_words\[\] = \{{(?P<body>.*?)\}};",
                source, re.S,
            )
            self.assertIsNotNone(match, name)
            if tensor["dtype"] == "float32":
                actual = re.findall(r"UINT32_C\(0x([0-9a-f]{8})\)",
                                    match.group("body"))
                self.assertEqual(actual, tensor["data"], name)
            else:
                actual = re.findall(r"INT64_MIN|-?INT64_C\([0-9]+\)",
                                    match.group("body"))
                expected = [
                    header_generator._i64_literal(
                        int.from_bytes(bytes.fromhex(encoded), "big", signed=True))
                    for encoded in tensor["data"]
                ]
                self.assertEqual(actual, expected, name)
            self.assertIn(f'"{name}",', source)
        for name in expectations:
            self.assertIn(f"{header_generator.tensor_symbol(name)}_words", source)


class PhiloxReferenceTests(unittest.TestCase):
    def test_random123_known_values(self) -> None:
        for counter, key, expected in generator.PHILOX_KNOWN_VECTORS:
            with self.subTest(counter=counter, key=key):
                self.assertEqual(generator.philox4x32_10(counter, key), expected)

    def test_state_word_mapping_uniform_and_counter_advance(self) -> None:
        counter, key, expected = generator.PHILOX_KNOWN_VECTORS[2]
        key64 = key[0] | (key[1] << 32)
        low64 = counter[0] | (counter[1] << 32)
        high64 = counter[2] | (counter[3] << 32)
        state = (1, generator._i64(key64), generator._i64(low64),
                 generator._i64(high64))
        words, next_state = generator.philox_lanes(state, 4)
        uniforms, uniform_state = generator.philox_uniforms(state, 4)
        self.assertEqual(words, expected)
        self.assertEqual(
            uniforms,
            tuple((word >> 8) * (2.0 ** -24) for word in expected),
        )
        self.assertEqual(uniform_state, next_state)
        self.assertEqual(next_state[2], generator._i64((low64 + 1) & generator.U64_MASK))
        self.assertEqual(next_state[3], generator._i64(high64 + (low64 == generator.U64_MASK)))

    def test_consumption_is_ceil_numel_over_four(self) -> None:
        initial = (1, 7, 19, 23)
        expected_low = {0: 19, 1: 20, 2: 20, 3: 20, 4: 20,
                        5: 21, 6: 21, 7: 21, 8: 21, 10: 22}
        for count, low in expected_low.items():
            with self.subTest(count=count):
                words, state = generator.philox_lanes(initial, count)
                self.assertEqual(len(words), count)
                self.assertEqual(state, (1, 7, low, 23))

    def test_signed_i64_word_mapping_boundaries(self) -> None:
        expected = {
            0: 0,
            (1 << 63) - 1: (1 << 63) - 1,
            1 << 63: -(1 << 63),
            generator.U64_MASK: -1,
        }
        for unsigned, signed in expected.items():
            with self.subTest(unsigned=unsigned):
                self.assertEqual(generator._i64(unsigned), signed)
                self.assertEqual(generator._u64(signed), unsigned)

    def test_split_calls_discard_unused_lanes(self) -> None:
        state = (1, 11, 0, 0)
        together, together_state = generator.philox_lanes(state, 2)
        first, split_state = generator.philox_lanes(state, 1)
        second, final_state = generator.philox_lanes(split_state, 1)
        self.assertEqual(first[0], together[0])
        self.assertNotEqual(second[0], together[1])
        self.assertEqual(together_state, (1, 11, 1, 0))
        self.assertEqual(final_state, (1, 11, 2, 0))

    def test_forward_backward_masks_regenerate_identically(self) -> None:
        state = (1, -7, 31, 5)
        forward, forward_state = generator.dropout_scales(state, 11, 0.25)
        backward, backward_state = generator.dropout_scales(state, 11, 0.25)
        self.assertEqual(backward, forward)
        self.assertEqual(backward_state, forward_state)
        self.assertEqual(set(forward), {0.0, generator._f32(4.0 / 3.0)})

    def test_inverted_scale_is_rounded_as_binary32(self) -> None:
        scales, _state = generator.dropout_scales((1, 0, 0, 0), 16, 0.25)
        kept = next(value for value in scales if value != 0.0)
        self.assertEqual(struct.pack(">f", kept).hex(), "3faaaaab")

    def test_chained_next_state_and_checkpoint_replay(self) -> None:
        initial = (1, generator._i64(0xFEDCBA9876543210), 101, 9)
        first, checkpoint = generator.dropout_scales(initial, 8, 0.25)
        second, final_state = generator.dropout_scales(checkpoint, 7, 0.25)
        replay, replay_state = generator.dropout_scales(tuple(checkpoint), 7, 0.25)
        self.assertEqual(replay, second)
        self.assertEqual(replay_state, final_state)
        self.assertNotEqual(first[:7], second)

    def test_positive_and_negative_zero_do_not_consume_state(self) -> None:
        state = (1, -1, -2, 17)
        for probability in (0.0, -0.0):
            with self.subTest(bits=probability.hex()):
                scales, next_state = generator.dropout_scales(state, 7, probability)
                self.assertEqual(scales, (1.0,) * 7)
                self.assertEqual(next_state, state)

    def test_max_counter_sentinel(self) -> None:
        maximum_minus_one = (1, 0, -2, -1)
        _words, maximum = generator.philox_lanes(maximum_minus_one, 1)
        self.assertEqual(maximum, (1, 0, -1, -1))
        with self.assertRaisesRegex(OverflowError, "counter overflow"):
            generator.philox_lanes(maximum, 1)
        words, unchanged = generator.philox_lanes(maximum, 0)
        self.assertEqual(words, ())
        self.assertEqual(unchanged, maximum)

    def test_counter_carry_and_overflow_are_explicit(self) -> None:
        words, state = generator.philox_lanes((1, 0, -1, 4), 1)
        self.assertEqual(len(words), 1)
        self.assertEqual(state, (1, 0, 0, 5))
        with self.assertRaisesRegex(OverflowError, "counter overflow"):
            generator.philox_lanes((1, 0, -1, -1), 1)

    def test_invalid_state_and_count_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "version 1"):
            generator.philox_lanes((2, 0, 0, 0), 1)
        with self.assertRaisesRegex(ValueError, "version 1"):
            generator.dropout_scales((2, 0, 0, 0), 1, 0.0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            generator.philox_lanes((1, 0, 0, 0), -1)
        with self.assertRaisesRegex(ValueError, "four counter"):
            generator.philox4x32_10((0, 0, 0), (0, 0))


@unittest.skipUnless(_pinned_torch_available(),
                     "PyTorch 2.13.0+cpu is required to generate N2 fixtures")
class N2GeneratedFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.encoded = encode_fixture(generator.build_payload())
        cls.payload = load_fixture(FROZEN)
        if cls.encoded != FROZEN.read_bytes():
            raise AssertionError("fresh N2 oracle generation differs from frozen bytes")

    def test_strict_source_lock_identity_and_case_operations(self) -> None:
        metadata = self.payload["generator"]
        self.assertEqual(metadata["framework"],
                         {"name": "pytorch", "version": "2.13.0+cpu"})
        self.assertEqual(metadata["name"], "n2.pytorch.primitives")
        self.assertEqual(metadata["version"], 1)
        self.assertEqual(metadata["seed"], 1729)
        self.assertEqual(metadata["source_sha256"],
                         hashlib.sha256(GENERATOR.read_bytes()).hexdigest())
        self.assertEqual(metadata["dependency_lock_sha256"],
                         hashlib.sha256(LOCK.read_bytes()).hexdigest())
        actual = {case["name"]: case["operation"] for case in self.payload["cases"]}
        self.assertEqual(actual, EXPECTED_CASE_OPERATIONS)

    def test_regeneration_is_byte_identical_in_fresh_processes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            outputs = [Path(temporary) / "first.json", Path(temporary) / "second.json"]
            for output in outputs:
                subprocess.run(
                    [sys.executable, str(GENERATOR), "--output", str(output)],
                    cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
                )
            self.assertEqual(outputs[0].read_bytes(), outputs[1].read_bytes())
            self.assertEqual(outputs[0].read_bytes(), self.encoded)

    def test_q0_roles_shapes_and_decoding(self) -> None:
        records = {record["name"]: record for record in self.payload["tensors"]}
        for case in self.payload["cases"]:
            for name in case["inputs"]:
                self.assertEqual(records[name]["role"], "input", name)
            expected_role = "analytic_gradient" if case["kind"] == "gradient" else "expected"
            for name in case["expectation"]["outputs"]:
                self.assertEqual(records[name]["role"], expected_role, name)
        expected_shapes = {
            "embedding.repeated.output.forward": [2, 3, 6],
            "embedding.repeated.gradient.dweight": [5, 6],
            "linear.bias.output.forward": [2, 3, 5],
            "linear.bias.gradient.dx": [2, 3, 4],
            "linear.bias.gradient.dweight": [5, 4],
            "linear.bias.gradient.dbias": [5],
            "linear.no_bias.output.forward": [2, 3, 5],
            "layer_norm.affine.output.forward": [2, 2, 4],
            "layer_norm.affine.gradient.dgamma": [4],
            "layer_norm.d1.output.forward": [1, 1, 1],
            "layer_norm.d1.gradient.dx": [1, 1, 1],
            "layer_norm.near_constant.output.forward": [1, 2, 4],
            "layer_norm.near_constant.gradient.dx": [1, 2, 4],
            "activation.gelu.output.forward": [1, 1, 8],
            "activation.relu.gradient.dx": [1, 1, 8],
            "residual.distinct.output.forward": [1, 3, 4],
            "dropout.train.p25.output.forward": [1, 2, 5],
            "dropout.train.p25.output.next_state": [4],
        }
        for name, shape in expected_shapes.items():
            record = tensor_by_name(self.payload, name)
            self.assertEqual(record["shape"], shape, name)
            count = 1
            for dimension in shape:
                count *= dimension
            self.assertEqual(len(decode_tensor(record)), count, name)

    def test_repeated_embedding_ids_accumulate_weight_gradient(self) -> None:
        ids = decode_tensor(tensor_by_name(self.payload, "embedding.repeated.input.ids"))
        self.assertEqual(ids.count(1), 2)
        self.assertEqual(ids.count(3), 2)
        gradient = decode_tensor(
            tensor_by_name(self.payload, "embedding.repeated.gradient.dweight"))
        self.assertEqual(gradient[2 * 6:3 * 6], (0.0,) * 6)
        self.assertTrue(any(value != 0.0 for value in gradient[1 * 6:2 * 6]))
        self.assertTrue(any(value != 0.0 for value in gradient[3 * 6:4 * 6]))

    def test_layer_norm_d1_and_near_constant(self) -> None:
        d1_output = decode_tensor(tensor_by_name(
            self.payload, "layer_norm.d1.output.forward"))
        d1_dx = decode_tensor(tensor_by_name(
            self.payload, "layer_norm.d1.gradient.dx"))
        d1_dgamma = decode_tensor(tensor_by_name(
            self.payload, "layer_norm.d1.gradient.dgamma"))
        d1_dbeta = decode_tensor(tensor_by_name(
            self.payload, "layer_norm.d1.gradient.dbeta"))
        self.assertEqual(d1_output, (-0.375,))
        self.assertEqual(d1_dx, (0.0,))
        self.assertEqual(d1_dgamma, (0.0,))
        self.assertEqual(d1_dbeta, (0.5,))
        near_output = decode_tensor(tensor_by_name(
            self.payload, "layer_norm.near_constant.output.forward"))
        near_dx = decode_tensor(tensor_by_name(
            self.payload, "layer_norm.near_constant.gradient.dx"))
        self.assertTrue(all(math.isfinite(value) for value in near_output + near_dx))
        self.assertTrue(any(value != 0.0 for value in near_dx))

    def test_gelu_tails_and_relu_signed_zero_contract(self) -> None:
        gelu = tensor_by_name(self.payload, "activation.gelu.output.forward")
        gelu_dx = tensor_by_name(self.payload, "activation.gelu.gradient.dx")
        self.assertEqual(gelu["data"][0], "80000000")
        self.assertEqual(gelu_dx["data"][0], "00000000")
        self.assertEqual(gelu["data"][-1], "41a00000")
        self.assertEqual(gelu_dx["data"][-1], "c0000000")
        # Exact GELU has derivative 1/2 at both signed-zero inputs.
        decoded_gelu = decode_tensor(gelu)
        decoded_gelu_dx = decode_tensor(gelu_dx)
        self.assertEqual(decoded_gelu[3:5], (-0.0, 0.0))
        self.assertEqual(decoded_gelu_dx[3:5], (0.375, 0.5))
        relu = tensor_by_name(self.payload, "activation.relu.output.forward")
        relu_dx = tensor_by_name(self.payload, "activation.relu.gradient.dx")
        self.assertEqual(relu["data"][3:5], ["00000000", "00000000"])
        self.assertEqual(relu_dx["data"][3:5], ["00000000", "00000000"])

    def test_repeated_residual_per_edge_and_accumulated_gradient(self) -> None:
        repeated_x = decode_tensor(tensor_by_name(
            self.payload, "residual.repeated.input.x"))
        repeated_branch = decode_tensor(tensor_by_name(
            self.payload, "residual.repeated.input.branch"))
        self.assertEqual(repeated_branch, repeated_x)
        upstream = decode_tensor(tensor_by_name(
            self.payload, "residual.repeated.input.upstream"))
        repeated_dx = decode_tensor(tensor_by_name(
            self.payload, "residual.repeated.gradient.dx"))
        repeated_dbranch = decode_tensor(tensor_by_name(
            self.payload, "residual.repeated.gradient.dbranch"))
        self.assertEqual(repeated_dx, upstream)
        self.assertEqual(repeated_dbranch, upstream)
        self.assertEqual(
            tuple(dx + dbranch for dx, dbranch in zip(repeated_dx, repeated_dbranch)),
            tuple(2.0 * value for value in upstream),
        )

    def test_dropout_golden_state_mask_and_modes(self) -> None:
        initial = decode_tensor(tensor_by_name(
            self.payload, "dropout.train.p25.input.state"))
        next_state = decode_tensor(tensor_by_name(
            self.payload, "dropout.train.p25.output.next_state"))
        self.assertEqual(next_state[:2], initial[:2])
        initial_counter = (initial[2] & generator.U64_MASK) | (
            (initial[3] & generator.U64_MASK) << 64)
        next_counter = (next_state[2] & generator.U64_MASK) | (
            (next_state[3] & generator.U64_MASK) << 64)
        self.assertEqual(next_counter - initial_counter, 3)
        p0_state = decode_tensor(tensor_by_name(
            self.payload, "dropout.train.p0.output.next_state"))
        self.assertEqual(p0_state, initial)
        for prefix in ("dropout.train.p0", "dropout.eval"):
            value = decode_tensor(tensor_by_name(self.payload, f"{prefix}.input.x"))
            output = decode_tensor(tensor_by_name(self.payload, f"{prefix}.output.forward"))
            self.assertEqual(output, value)
        p25_output = decode_tensor(tensor_by_name(
            self.payload, "dropout.train.p25.output.forward"))
        p25_dx = decode_tensor(tensor_by_name(
            self.payload, "dropout.train.p25.gradient.dx"))
        self.assertTrue(any(value == 0.0 for value in p25_output))
        self.assertTrue(any(value != 0.0 for value in p25_dx))


if __name__ == "__main__":
    unittest.main()
