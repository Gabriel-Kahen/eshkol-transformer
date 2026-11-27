"""Check the MOD5 C2 input artifacts and independent D2 cursor plan."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tests.d2.reference import ReferenceConfig, ReferenceDataset, decode_reference_cursor
from tests.d2.test_resources import load_d1_resource, write_d1_resource
from tests.mod5 import mod5_v256_c2_fixture as fixture
from tests.t1.reference import Identity, decode_artifact, encode_artifact, encode_bytes
from tests.x1 import manifest_format as x1


class Mod5V256C2FixtureTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix="mod5-v256-c2-fixture-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "inputs"
        self.record = fixture.build(self.root)

    def open_reference_dataset(self):
        config = ReferenceConfig(**self.record["d2"])
        tokenizer = decode_artifact((self.root / "tokenizer.etok").read_bytes())
        corpus = self.root / config.directory
        shards, trailer = load_d1_resource(
            corpus, expected_fingerprint=tokenizer.fingerprint, expected_vocab=256,
        )
        dataset = ReferenceDataset(
            shards, config, manifest_digest=trailer,
            tokenizer_fingerprint=tokenizer.fingerprint, vocab_size=256,
        )
        return dataset, tokenizer, corpus, shards

    def test_build_and_verify_are_reproducible(self):
        second = self.root.with_name("repeat")
        self.assertEqual(self.record, fixture.build(second))
        self.assertEqual(self.record, fixture.verify(self.root))
        inputs = {path.relative_to(self.root): path.read_bytes()
                  for path in self.root.rglob("*") if path.is_file()}
        repeated = {path.relative_to(second): path.read_bytes()
                    for path in second.rglob("*") if path.is_file()}
        self.assertEqual(inputs, repeated)
        with self.assertRaises(FileExistsError):
            fixture.build(self.root)

    def test_canonical_x1_and_exact_full_mask_cursor_suffix(self):
        canonical = (self.root / "expected-x1.json").read_bytes()
        resolved = x1.load_manifest_bytes(canonical)
        self.assertEqual(resolved["resolved"], json.loads((self.root / "config.json").read_bytes()))
        self.assertEqual(resolved["resolved"], {
            "config-schema-major": 1,
            "config-schema-minor": 0,
            "model.context-length": 8,
            "model.device": "cpu",
            "model.dtype": "f32",
            "model.head-size": 4,
            "model.hidden-size": 16,
            "model.kv-head-count": 2,
            "model.layer-count": 1,
            "model.query-head-count": 4,
            "model.vocabulary-size": 256,
            "run.deterministic": True,
            "run.seed": fixture.PHILOX_KEY,
            "training.accumulation-steps": 1,
        })
        self.assertEqual(x1.fingerprint(canonical), self.record["x1_fingerprint"])

        producer, tokenizer, corpus, shards = self.open_reference_dataset()
        self.assertEqual(tokenizer.identity, Identity("raw"))
        self.assertEqual(tokenizer.fingerprint, self.record["tokenizer_fingerprint"])
        self.assertFalse(producer.config.packing)
        self.assertIsNone(producer.config.shuffle_seed)
        self.assertEqual(self.record["prefix_updates"], 2)
        self.assertEqual(self.record["total_updates"], 6)
        self.assertEqual(len(shards), 6)
        self.assertEqual(len(set(shards)), 6)
        cursor_hashes = self.record["oracle_cursor_sha256_by_next_ordinal"]
        self.assertEqual(len(cursor_hashes), 7)
        epoch_start = producer.snapshot()
        for ordinal, tokens in enumerate(shards):
            self.assertEqual(hashlib.sha256(producer.snapshot()).hexdigest(), cursor_hashes[ordinal])
            if ordinal == 2:
                current = producer.snapshot()
            original = (corpus / f"doc-{ordinal}.bin").read_bytes()
            self.assertEqual(len(original), 9)
            self.assertEqual(tokens, encode_bytes(tokenizer.identity, original))
            batch = producer.next_batch()
            self.assertEqual(batch.inputs, (tuple(original[:-1]),))
            self.assertEqual(batch.targets, (tuple(original[1:]),))
            self.assertEqual(batch.loss_mask, ((True,) * 8,))
        final = producer.snapshot()
        self.assertEqual(hashlib.sha256(final).hexdigest(), cursor_hashes[6])
        self.assertIsNone(producer.next_batch())
        self.assertEqual(producer.snapshot(), final)

        epoch_fields = decode_reference_cursor(epoch_start)
        current_fields = decode_reference_cursor(current)
        self.assertEqual(epoch_fields.next_ordinal, 0)
        self.assertEqual(current_fields.next_ordinal, 2)
        self.assertEqual(current_fields.manifest_digest.hex(), self.record["manifest_trailer_sha256"])
        self.assertEqual(current[:120], epoch_start[:120])
        self.assertEqual(current[128:-32], epoch_start[128:-32])
        receiver, _, _, _ = self.open_reference_dataset()
        receiver.seek(current)
        for tokens in shards[2:]:
            self.assertEqual(receiver.next_batch().targets, (tokens[1:],))
        self.assertEqual(receiver.snapshot(), final)
        receiver.seek(epoch_start)
        self.assertEqual(receiver.snapshot(), epoch_start)

    def test_valid_checksummed_corpus_cannot_replace_original_documents(self):
        _, tokenizer, corpus, shards = self.open_reference_dataset()
        replacement = (100,) + shards[0][1:]
        write_d1_resource(
            corpus, (replacement,) + shards[1:],
            fingerprint=tokenizer.fingerprint, vocab=256,
        )
        self.assertEqual(self.open_reference_dataset()[3][0], replacement)
        with self.assertRaisesRegex(ValueError, "T1 encodings"):
            fixture.verify(self.root)

    def test_valid_t1_artifact_with_foreign_identity_is_rejected(self):
        (self.root / "tokenizer.etok").write_bytes(encode_artifact(Identity("strict")))
        with self.assertRaisesRegex(ValueError, "canonical V256 T1 raw"):
            fixture.verify(self.root)

    def test_valid_canonical_x1_with_foreign_seed_is_rejected(self):
        source = self.root / "config.json"
        canonical = self.root / "expected-x1.json"
        config = json.loads(source.read_bytes())
        manifest = x1.load_manifest_bytes(canonical.read_bytes())
        config["run.seed"] += 1
        manifest["resolved"] = config
        source.write_bytes(x1.canonical_bytes(config))
        canonical.write_bytes(x1.canonical_bytes(manifest))
        self.assertEqual(x1.load_manifest_bytes(canonical.read_bytes())["resolved"], config)
        with self.assertRaisesRegex(ValueError, "fixed MOD5 V256 profile"):
            fixture.verify(self.root)

    def test_stale_cursor_plan_is_rejected(self):
        self.record["oracle_cursor_sha256_by_next_ordinal"].reverse()
        (self.root / "fixture.json").write_bytes(x1.canonical_bytes(self.record))
        with self.assertRaisesRegex(ValueError, "record differs"):
            fixture.verify(self.root)


if __name__ == "__main__":
    unittest.main()
