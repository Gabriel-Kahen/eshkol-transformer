"""Drive D1, live AOT training, C1, and IO5 restores."""

from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools/io5"
sys.path[:0] = [str(TOOLS), str(ROOT / "tests/c1"), str(ROOT / "tests/io5")]
import gguf_interop as gg
import quantized_interchange as quant
import safetensors_interop as safe
from gguf_wire_fixture import unpack_quant_gguf
from test_checkpoint_format import build_container

PROFILE = "eshkol-mod5-v1"
PROVIDER = "i2-dense-cpu-f32-v1"
BRIDGE_ENV = "MOD5_E2E_AOT"
INDEX = "model.safetensors.index.json"
SHARD = "shard-0000000000000000.ets"
CORPUS = (1, 4, 1, 5, 9, 2, 6, 5, 3)
ROSTER = {
    "blocks.0.attention.key.weight": (8, 16),
    "blocks.0.attention.output.weight": (16, 16),
    "blocks.0.attention.query.weight": (16, 16),
    "blocks.0.attention.value.weight": (8, 16),
    "blocks.0.ffn.down.weight": (16, 32),
    "blocks.0.ffn.gate.weight": (32, 16),
    "blocks.0.ffn.up.weight": (32, 16),
    "blocks.0.norm1.weight": (16,),
    "blocks.0.norm2.weight": (16,),
    "head.weight": (16, 16),
    "norm_final.weight": (16,),
    "token_embedding.weight": (16, 16),
}
GGUF_NAMES = (
    "blk.0.attn_k.weight", "blk.0.attn_output.weight", "blk.0.attn_q.weight",
    "blk.0.attn_v.weight", "blk.0.ffn_down.weight", "blk.0.ffn_gate.weight",
    "blk.0.ffn_up.weight", "blk.0.attn_norm.weight", "blk.0.ffn_norm.weight",
    "output.weight", "output_norm.weight", "token_embd.weight",
)
Q8_NAME = "blk.0.ffn_down.weight"
Q8_PATH = ("blocks", "0", "ffn", "down", "weight")
EVAL = re.compile(
    r"EVAL loss=(\S+) loss_f64=(0x[0-9a-f]+(?:\.[0-9a-f]+)?p[+-]\d+) "
    r"logits_f32_sha256=([0-9a-f]{64})"
)
LIMITS = safe.Limits()


def flip(raw, offset):
    return raw[:offset] + bytes([raw[offset] ^ 1]) + raw[offset + 1:]


def resign(raw):
    return raw[:-32] + hashlib.sha256(raw[:-32]).digest()


class Mod5EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        binary = os.environ.get(BRIDGE_ENV)
        if binary is None:
            if __name__ == "__main__":
                raise RuntimeError(f"Set {BRIDGE_ENV} through scripts/test-mod5-e2e-io5.sh")
            raise unittest.SkipTest("requires the compiled MOD5 E2E binary")
        cls.binary = Path(binary).resolve()
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        cls.directory = Path(directory.name)
        cls.corpus = cls.directory / "corpus"
        cls.corpus.mkdir()
        cls.run_command(cls.binary, "--mode", "write-corpus", cls.corpus)
        raw = (cls.corpus / SHARD).read_bytes()
        offset = struct.unpack_from("<I", raw, 12)[0]
        if struct.unpack_from("<9Q", raw, offset) != CORPUS:
            raise AssertionError("unexpected D1 corpus tokens")
        cls.originals = {}
        for variant in ("untied", "tied"):
            path = cls.directory / f"{variant}.etcp"
            options = ("--tied",) if variant == "tied" else ()
            first = cls.run_command(cls.binary, "--mode", "train-save", path, *options)
            duplicate = cls.directory / f"{variant}-repeat.etcp"
            second = cls.run_command(cls.binary, "--mode", "train-save", duplicate, *options)
            if first.stdout != second.stdout or path.read_bytes() != duplicate.read_bytes():
                raise AssertionError(f"{variant} training and checkpoint were not repeatable")
            cls.originals[variant] = (path, first.stdout)
            print(f"MOD5 {variant} C1 sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")
            print(first.stdout, end="", flush=True)

    @classmethod
    def run_command(cls, *args, status=0, cwd=None):
        command = list(map(str, args))
        result = subprocess.run(
            command, cwd=cls.directory if cwd is None else cwd,
            timeout=60, text=True, capture_output=True,
        )
        if result.returncode != status or (status == 0 and result.stderr):
            raise AssertionError(
                f"{command} returned {result.returncode}, expected {status}\n"
                f"{result.stdout}{result.stderr}"
            )
        return result

    def cli(self, *args, status=0):
        return self.run_command(sys.executable, TOOLS / "model_mapping.py", *args, status=status)

    def quant_cli(self, *args, status=0):
        return self.run_command(sys.executable, TOOLS / "quantized_interchange.py", *args, status=status)

    def train_eval(self, variant):
        path, stdout = self.originals[variant]
        lines = stdout.splitlines()
        self.assertEqual(lines[0], "GRADIENTS 11" if variant == "tied" else "GRADIENTS 12")
        self.assertEqual(len(lines), 15)
        losses = []
        for step, line in enumerate(lines[1:-1]):
            label, index, loss = line.split()
            self.assertEqual((label, index), ("LOSS", str(step)))
            losses.append(float(loss))
        self.assertTrue(all(math.isfinite(loss) and loss >= 0 for loss in losses))
        self.assertTrue(all(right < left for left, right in zip(losses, losses[1:])))
        evaluation = EVAL.fullmatch(lines[-1])
        self.assertIsNotNone(evaluation)
        loss, bits, _ = evaluation.groups()
        self.assertEqual(losses[-1], float(loss))
        self.assertEqual(float.fromhex(bits), float(loss))
        return path, lines[-1] + "\n"

    def assert_live_eval(self, path, expected):
        self.assertEqual(
            self.run_command(self.binary, "--mode", "load-eval", path).stdout, expected,
        )

    def assert_live_rejected(self, path, diagnostic, **options):
        result = self.run_command(self.binary, "--mode", "load-eval", path, status=2, **options)
        self.assertIn("MOD5 E2E ERROR", result.stdout)
        self.assertIn(diagnostic, result.stdout)
        self.assertNotIn("EVAL loss=", result.stdout)

    def assert_io5_rejected(self, route, wire, destination, diagnostic, *options):
        result = self.cli(f"{route}-to-c1", wire, destination, *options, status=2)
        self.assertIn(diagnostic, result.stderr)
        self.assertFalse(destination.exists())

    def test_five_routes_for_each_trained_variant(self):
        for variant in self.originals:
            with self.subTest(variant=variant):
                original, evaluation = self.train_eval(variant)
                original_bytes = original.read_bytes()
                state = safe.decode_c1(original_bytes)
                self.assertEqual(state.provider, PROVIDER)
                self.assertTrue(all(t.kind == "parameter" for t in state.tensors))
                self.assertEqual({".".join(t.path): t.shape for t in state.tensors}, ROSTER)
                head, embedding = state.tensors[9], state.tensors[11]
                if variant == "tied":
                    self.assertEqual(state.aliases, ((9, 11),))
                    self.assertEqual(head.data, embedding.data)
                else:
                    self.assertEqual(state.aliases, ())
                    self.assertNotEqual(head.data, embedding.data)
                self.assert_live_eval(original, evaluation)
                variants = self.directory / variant
                variants.mkdir()
                routes = [
                    ("mapped-hf", "mapped-safetensors", "huggingface", False),
                    ("external-hf", "mapped-safetensors", "huggingface", True),
                    ("sharded-hf", "sharded-safetensors", "huggingface", False),
                    ("mapped-gguf", "mapped-gguf", "gguf", False),
                    ("external-gguf", "mapped-gguf", "gguf", True),
                ]
                for label, route, scheme, external in routes:
                    with self.subTest(route=label):
                        directory = variants / label
                        directory.mkdir()
                        extension = "gguf" if scheme == "gguf" else "safetensors"
                        wire = directory / (INDEX if label == "sharded-hf" else f"mapped.{extension}")
                        selection = ("--profile", PROFILE, "--scheme", scheme)
                        limits = ("--max-shard-bytes", "3000") if label == "sharded-hf" else ()
                        self.cli(f"c1-to-{route}", original, wire, *selection, *limits)
                        options = ()
                        if external:
                            codec = gg if scheme == "gguf" else safe
                            decode, encode = (
                                (codec.decode_gguf_tensors, codec.encode_gguf_tensors)
                                if scheme == "gguf" else
                                (codec.decode_safetensors_tensors, codec.encode_safetensors_tensors)
                            )
                            named, _ = decode(wire.read_bytes(), LIMITS)
                            wire = directory / f"external.{extension}"
                            wire.write_bytes(encode(named, {}, LIMITS))
                            options = (*selection, "--external", "--provider", PROVIDER)
                            if variant == "tied":
                                options += ("--tie-head-to-embedding",)
                        restored = directory / "restored.etcp"
                        self.cli(f"{route}-to-c1", wire, restored, *options)
                        self.assertEqual(restored.read_bytes(), original_bytes)
                        self.assert_live_eval(restored, evaluation)
                        if label == "sharded-hf":
                            index = json.loads(wire.read_bytes())
                            shards = {directory / name for name in index["weight_map"].values()}
                            self.assertGreater(len(shards), 1)
                            self.assertTrue(all(path.stat().st_size <= 3000 for path in shards))
                            shard = sorted(shards)[0]
                            raw = shard.read_bytes()
                            shard.write_bytes(flip(raw, 8 + struct.unpack_from("<Q", raw)[0]))
                            self.assert_io5_rejected(
                                route, wire, directory / "rejected.etcp", "shard SHA-256 mismatch",
                            )
                        if label == "mapped-hf":
                            raw = wire.read_bytes()
                            wire.write_bytes(flip(raw, 8 + struct.unpack_from("<Q", raw)[0]))
                            self.assert_io5_rejected(
                                route, wire, directory / "rejected.etcp", "checksum",
                            )
                        if label == "mapped-gguf":
                            self.assertTrue(all(len(t.data) % gg.ALIGNMENT == 0 for t in state.tensors))
                            raw = wire.read_bytes()
                            payload_start = len(raw) - sum(len(t.data) for t in state.tensors)
                            wire.write_bytes(flip(raw, payload_start))
                            self.assert_io5_rejected(
                                route, wire, directory / "rejected.etcp", "checksum",
                            )
                        print(f"MOD5 {variant} {label} bytes and fresh AOT evaluation: PASS", flush=True)

    def test_narrow_safetensors_fresh_eval_for_each_trained_variant(self):
        for variant in self.originals:
            original, evaluation = self.train_eval(variant)
            baseline = EVAL.fullmatch(evaluation.rstrip("\n"))
            original_bytes = original.read_bytes()
            for dtype, weight_limit, loss_limit in (("F16", 0.001, 0.002), ("BF16", 0.008, 0.010)):
                with self.subTest(variant=variant, dtype=dtype):
                    prefix = f"{variant}-{dtype}"
                    wire = self.directory / f"{prefix}.safetensors"
                    restored = self.directory / f"{prefix}.etcp"
                    command = (sys.executable, TOOLS / "quantized_interchange.py")
                    self.run_command(*command, "c1-to-safetensors", original, wire, "--dtype", dtype,
                                     "--profile", PROFILE)
                    self.run_command(*command, "safetensors-to-c1", wire, restored, "--profile", PROFILE)

                    restored_bytes = restored.read_bytes()
                    self.assertEqual(safe.decode_c1(restored_bytes).provider, PROVIDER)
                    metrics = quant.measure_reconstruction_error(original_bytes, restored_bytes)
                    weight_delta = metrics["__overall__"]["max_abs"]
                    self.assertGreater(weight_delta, 0)
                    self.assertLessEqual(weight_delta, weight_limit)

                    result = self.run_command(self.binary, "--mode", "load-eval", restored)
                    restored_eval = EVAL.fullmatch(result.stdout.rstrip("\n"))
                    self.assertIsNotNone(restored_eval)
                    new_loss = float.fromhex(restored_eval[2])
                    self.assertTrue(math.isfinite(new_loss))
                    self.assertEqual(new_loss, float(restored_eval[1]))
                    self.assertNotEqual(restored_eval[3], baseline[3])
                    loss_delta = abs(new_loss - float.fromhex(baseline[2]))
                    self.assertLessEqual(loss_delta, loss_limit)
                    self.assert_live_eval(restored, result.stdout)
                    print(f"MOD5 {prefix} weight_max_abs={weight_delta:.9g} loss_abs={loss_delta:.9g} "
                          f"logits_f32_sha256={restored_eval[3]}", flush=True)

    def inspect_q8_wire(self, raw, state):
        metadata, rows, prefix_end, data_start = unpack_quant_gguf(raw)
        self.assertEqual(tuple(row[0] for row in rows), GGUF_NAMES)
        self.assertEqual(metadata["general.file_type"], (gg.UINT32, 7))
        self.assertEqual(metadata["general.quantization_version"], (gg.UINT32, 2))
        self.assertEqual(metadata["eshkol.io5.profile"], (gg.STRING, "eshkol-decoder-v1"))
        self.assertEqual(metadata["eshkol.io5.scheme"], (gg.STRING, "gguf"))
        self.assertEqual(metadata["eshkol.io5.q8_fallback_dtype"], (gg.STRING, "F32"))
        self.assertEqual(json.loads(metadata["eshkol.io5.tensor_dtypes"][1]), {
            name: "Q8_0" if name == Q8_NAME else "F32" for name in GGUF_NAMES
        })
        self.assertEqual(json.loads(metadata["eshkol.io5.manifest"][1]), {
            "provider": state.provider,
            "entries": [
                {"name": name, "path": list(tensor.path), "kind": tensor.kind}
                for name, tensor in zip(GGUF_NAMES, state.tensors, strict=True)
            ],
            "aliases": [list(group) for group in state.aliases],
        })
        self.assertEqual(data_start % 32, 0)
        self.assertEqual(raw[prefix_end:data_start], bytes(data_start - prefix_end))
        cursor = 0
        restored = []
        q8_byte = None
        weight_delta = 0.0
        for tensor, (name, dims, kind, offset, data) in zip(state.tensors, rows, strict=True):
            self.assertEqual(dims, tensor.shape[::-1])
            self.assertEqual(offset, cursor)
            padding = (-len(data)) % 32
            cursor += len(data) + padding
            self.assertEqual(raw[data_start + cursor - padding:data_start + cursor], bytes(padding))
            if name == Q8_NAME:
                self.assertEqual((tensor.path, dims, kind, len(data)), (Q8_PATH, (32, 16), 8, 544))
                q8_byte = data_start + offset + 2
                blocks = []
                for (scale, *quants), weights in zip(
                    struct.iter_unpack("<e32b", data), struct.iter_unpack("<32f", tensor.data), strict=True,
                ):
                    amax = max(map(abs, weights))
                    self.assertEqual(struct.pack("<e", scale), struct.pack("<e", amax / 127))
                    self.assertTrue(all(-127 <= quant <= 127 for quant in quants))
                    values = tuple(scale * quant for quant in quants)
                    error = max(abs(a - b) for a, b in zip(weights, values, strict=True))
                    self.assertLessEqual(error, max(scale / 2, amax - 127 * scale))
                    weight_delta = max(weight_delta, error)
                    blocks.append(struct.pack("<32f", *values))
                data = b"".join(blocks)
            else:
                self.assertEqual(kind, 0)
                self.assertEqual(data, tensor.data)
            restored.append(replace(tensor, data=data))
        self.assertEqual(cursor, 9952)
        self.assertEqual(len(raw), data_start + cursor)
        self.assertIsNotNone(q8_byte)
        return replace(state, tensors=tuple(restored)), q8_byte, weight_delta

    def test_trained_q8_gguf_fresh_scoring(self):
        for variant in self.originals:
            with self.subTest(variant=variant):
                original, baseline = self.train_eval(variant)
                self.assert_live_eval(original, baseline)
                original_bytes = original.read_bytes()
                state = safe.decode_c1(original_bytes)
                self.assertEqual(state.provider, PROVIDER)
                self.assertEqual({".".join(t.path): t.shape for t in state.tensors}, ROSTER)
                self.assertEqual(state.aliases, ((9, 11),) if variant == "tied" else ())
                directory = self.directory / f"{variant}-q8"
                directory.mkdir()
                wire = directory / "trained.gguf"
                self.quant_cli(
                    "c1-to-gguf", original, wire, "--profile", "eshkol-decoder-v1",
                    "--scheme", "gguf", "--dtype", "Q8_0", "--q8-fallback", "F32",
                )
                raw = wire.read_bytes()
                expected, q8_byte, weight_delta = self.inspect_q8_wire(raw, state)
                restored = directory / "restored.etcp"
                self.quant_cli("gguf-to-c1", wire, restored)
                self.assertEqual(safe.decode_c1(restored.read_bytes()), expected)
                self.assertGreater(weight_delta, 0)

                score = self.run_command(self.binary, "--mode", "load-eval", restored).stdout
                self.assert_live_eval(restored, score)
                result = EVAL.fullmatch(score.rstrip("\n"))
                self.assertIsNotNone(result)
                loss, bits, logits = result.groups()
                self.assertTrue(math.isfinite(float(loss)) and float(loss) >= 0)
                self.assertEqual(float.fromhex(bits), float(loss))
                _, baseline_bits, baseline_logits = EVAL.fullmatch(baseline.rstrip("\n")).groups()
                loss_delta = abs(float.fromhex(bits) - float.fromhex(baseline_bits))
                self.assertGreater(loss_delta, 0)
                self.assertLessEqual(loss_delta, 0.01)
                self.assertNotEqual(logits, baseline_logits)

                corrupt = directory / "corrupt.gguf"
                corrupt.write_bytes(flip(raw, q8_byte))
                rejected = directory / "rejected.etcp"
                diagnostic = "IO5 quantized GGUF payload SHA-256 mismatch"
                result = self.quant_cli("gguf-to-c1", corrupt, rejected, status=2)
                self.assertIn(diagnostic, result.stderr)
                self.assertFalse(rejected.exists())
                existing = directory / "existing.etcp"
                existing.write_bytes(original_bytes)
                result = self.quant_cli("gguf-to-c1", corrupt, existing, "--overwrite", status=2)
                self.assertIn(diagnostic, result.stderr)
                self.assertEqual(existing.read_bytes(), original_bytes)
                print(
                    f"MOD5 {variant} mixed Q8_0 fresh evaluation: {score.strip()} "
                    f"max_weight_delta={weight_delta:.9g} mean_loss_delta={loss_delta:.9g}",
                    flush=True,
                )

    def test_external_roster_and_live_c1_tamper(self):
        original, evaluation = self.train_eval("untied")
        directory = self.directory / "tamper"
        directory.mkdir()
        raw = original.read_bytes()
        state = safe.decode_c1(raw)
        payload = struct.unpack_from("<Q", raw, 64)[0]
        wire = directory / "mapped.safetensors"
        self.cli("c1-to-mapped-safetensors", original, wire, "--profile", PROFILE, "--scheme", "huggingface")
        named, _ = safe.decode_safetensors_tensors(wire.read_bytes(), LIMITS)
        bias = safe.Tensor(("bias",), "parameter", (16,), bytes(64))
        gate_path = tuple("blocks.0.ffn.gate.weight".split("."))
        first = state.tensors[0]
        options = ("--external", "--profile", PROFILE, "--scheme", "huggingface", "--provider", PROVIDER)
        for label, changed_wire, changed_tensors, diagnostic in (
            ("bias", named | {"model.layers.0.self_attn.q_proj.bias": bias},
             tuple(sorted(
                 state.tensors + (replace(bias, path=("blocks", "0", "attention", "query", "bias")),),
                 key=lambda tensor: tensor.path,
             )), "unmapped external block name"),
            ("missing-gate", {name: tensor for name, tensor in named.items()
                              if name != "model.layers.0.mlp.gate_proj.weight"},
             tuple(tensor for tensor in state.tensors if tensor.path != gate_path),
             "missing MOD5 block tensor"),
        ):
            with self.subTest(tamper=label):
                external = directory / f"{label}.safetensors"
                external.write_bytes(safe.encode_safetensors_tensors(changed_wire, {}, LIMITS))
                self.assert_io5_rejected(
                    "mapped-safetensors", external, directory / f"{label}-rejected.etcp", diagnostic, *options,
                )
                invalid = directory / f"{label}.etcp"
                invalid.write_bytes(safe.encode_c1(replace(state, tensors=changed_tensors)))
                self.assert_live_rejected(invalid, "unexpected MOD5 parameter roster")

        corrupt = directory / "stale-checksum.etcp"
        corrupt.write_bytes(flip(raw, payload))
        self.assert_live_rejected(corrupt, "checksum")

        bad_shape = directory / "wrong-shape.etcp"
        bad_shape.write_bytes(safe.encode_c1(replace(state, tensors=(
            replace(first, shape=(16, 8)), *state.tensors[1:],
        ))))
        self.assert_live_rejected(bad_shape, "unexpected MOD5 parameter metadata")

        # A different, correctly checksummed finite model is valid and must
        # produce different live results. This also detects a canned EVAL path.
        changed = list(state.tensors)
        head = changed[9]
        changed[9] = replace(head, data=struct.pack("<f", 1.0) + head.data[4:])
        different = directory / "different-model.etcp"
        different.write_bytes(safe.encode_c1(replace(state, tensors=tuple(changed))))
        observed = self.run_command(self.binary, "--mode", "load-eval", different).stdout
        self.assertRegex(observed.strip(), EVAL)
        self.assertNotEqual(observed, evaluation)

    def test_tied_live_c1_rejects_disagreeing_payloads(self):
        original, _ = self.originals["tied"]
        state = safe.decode_c1(original.read_bytes())
        tensors = list(state.tensors)
        tensors[9] = replace(tensors[9], data=flip(tensors[9].data, 0))
        entries = tuple(
            (tuple(segment.encode() for segment in tensor.path), 1, 3, tensor.shape, tensor.data)
            for tensor in tensors
        )
        # The public encoder rejects this alias before writing, so use the
        # independent C1 fixture builder to exercise live load admission.
        invalid = self.directory / "mismatched-tie.etcp"
        invalid.write_bytes(build_container(entries, state.aliases, state.provider.encode()))
        self.assert_live_rejected(invalid, "tied parameter payloads conflict")

    def test_d1_inputs_are_consumed_and_validated(self):
        original, evaluation = self.train_eval("untied")
        directory = self.directory / "bad-corpus"
        target = directory / "corpus"
        shutil.copytree(self.corpus, target)
        shard, manifest = target / SHARD, target / "manifest.etm"
        raw, manifest_raw = shard.read_bytes(), manifest.read_bytes()
        offset = struct.unpack_from("<I", raw, 12)[0]
        shard.write_bytes(flip(raw, offset))
        self.assert_live_rejected(original, "SHA-256", cwd=directory)

        digest_offset = struct.unpack_from("<I", manifest_raw, 12)[0] + 24

        def replace_first_token(token):
            updated = resign(raw[:offset] + struct.pack("<Q", token) + raw[offset + 8:])
            shard.write_bytes(updated)
            manifest.write_bytes(resign(
                manifest_raw[:digest_offset] + updated[-32:] + manifest_raw[digest_offset + 32:]
            ))

        replace_first_token(2)
        changed = self.run_command(self.binary, "--mode", "load-eval", original, cwd=directory).stdout
        self.assertRegex(changed.strip(), EVAL)
        self.assertNotEqual(changed, evaluation)

        replace_first_token(16)
        self.assert_live_rejected(original, "token outside", cwd=directory)


if __name__ == "__main__":
    unittest.main()
