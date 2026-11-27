"""Compiler-free T1/D1/C1 provenance and swap checks for the V=256 fixture."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tests.eval5 import mod5_v256_fixture as fixture


class Mod5V256FixtureTests(unittest.TestCase):
    def test_exact_artifacts_and_reference_score(self):
        with TemporaryDirectory(prefix="eval5-v256-") as temp:
            root = Path(temp) / "fixture"
            record = fixture.build(root)
            self.assertEqual(fixture.verify(root), record)
            self.assertEqual(record["tokenizer_fingerprint"],
                             "sha256:eshkol-byte-tokenizer-v1:"
                             "aabb31f49216963582383a00fd2e85d7c20b658d5bef0e7945344ed6bfe50704")
            self.assertEqual(record["tokenizer_artifact_sha256"],
                             "62e2aa0feef161ec7115f50f172bc24b3b4c51847150d6a663ab71ce9cb001cd")
            self.assertEqual(record["checkpoint_sha256"],
                             "505358be27f676998816c9d2ed194358edb017be4ccb2050069c1e22d9279eae")
            self.assertEqual(record["splits"]["train"]["dataset_sha256"],
                             "71e4c68f07866a317aa091575ef555ecda37c04347381dbc795adcb5371023b6")
            self.assertEqual(record["splits"]["eval-clean"]["dataset_sha256"],
                             "b2e31837e5094b79d95c5ac30a799a54e1c117f60e12511d3853f5ef5bfd69e2")
            self.assertEqual(record["splits"]["train"]["ordered_shard_sha256"], [
                "fbd3dd3e1b110ead04fa6845fc2c4a31fc90db4cc08addd5e684617e36fbea8b",
                "6cc666e5d59082005ff87ba0a31dd4d7bb2042c9958e9af8b71808a818cb50aa",
                "24076e7603beef90f88506b35b4cec332ed6e26953a04b8d0cfda8a5bb366a09",
            ])
            self.assertEqual(record["splits"]["eval-clean"]["ordered_shard_sha256"], [
                "30ecf9db79188bf872f47cf224d2a68e7843bcc07fb95666d42b16cb1cd97095",
                "6681375b2ce7a417e087c734b599a43d6457fdd7a78bb4f4e1edfc1520b3540b",
                "d65a043910a29467238043153d8b5eb639d71c2bedc781c8a939fd6433238816",
            ])
            self.assertEqual(record["score_reference"]["audit"]["matched_eval_ngrams"], 0)
            self.assertEqual(record["score_reference"]["lm"]["byte_count"], 16)
            self.assertAlmostEqual(record["score_reference"]["lm"]["total_nll"],
                                   90.78076731811535, places=8)
            self.assertEqual(record["score_reference"]["mc"]["correct_count"], 2)

    def test_rejects_swapped_split_and_shard(self):
        with TemporaryDirectory(prefix="eval5-v256-") as temp:
            root = Path(temp) / "fixture"
            fixture.build(root)
            train = root / "train"
            evaluation = root / "eval-clean"
            temporary = root / "swap"
            train.rename(temporary)
            evaluation.rename(train)
            temporary.rename(evaluation)
            with self.assertRaises(ValueError):
                fixture.verify(root)

        with TemporaryDirectory(prefix="eval5-v256-") as temp:
            root = Path(temp) / "fixture"
            fixture.build(root)
            shard0 = root / "train" / "shard-0000000000000000.ets"
            shard1 = root / "train" / "shard-0000000000000001.ets"
            first, second = shard0.read_bytes(), shard1.read_bytes()
            shard0.write_bytes(second)
            shard1.write_bytes(first)
            with self.assertRaises(ValueError):
                fixture.verify(root)

    def test_rejects_document_or_record_swap(self):
        with TemporaryDirectory(prefix="eval5-v256-") as temp:
            root = Path(temp) / "fixture"
            fixture.build(root)
            doc = root / "eval-clean" / "doc-0.bin"
            doc.write_bytes(doc.read_bytes()[::-1])
            with self.assertRaises(ValueError):
                fixture.verify(root)

        with TemporaryDirectory(prefix="eval5-v256-") as temp:
            root = Path(temp) / "fixture"
            fixture.build(root)
            record = root / "fixture.json"
            record.write_bytes(record.read_bytes().replace(
                b"71e4c68f07866a317aa091575ef555ecda37c04347381dbc795adcb5371023b6",
                b"b2e31837e5094b79d95c5ac30a799a54e1c117f60e12511d3853f5ef5bfd69e2"))
            with self.assertRaises(ValueError):
                fixture.verify(root)

    def test_rejects_changed_checkpoint_or_manifest_bytes(self):
        with TemporaryDirectory(prefix="eval5-v256-") as temp:
            root = Path(temp) / "fixture"
            fixture.build(root)
            for path in (root / "model.etcp", root / "eval-clean" / "manifest.etm"):
                original = path.read_bytes()
                changed = bytearray(original)
                changed[-1] ^= 1
                path.write_bytes(changed)
                with self.subTest(path=path), self.assertRaises(ValueError):
                    fixture.verify(root)
                path.write_bytes(original)
            self.assertEqual(fixture.verify(root)["scope"], "development-reference-only")


if __name__ == "__main__":
    unittest.main()
