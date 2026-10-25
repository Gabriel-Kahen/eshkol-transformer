"""Source contract for the trained-C2 EOS reference; no compiled admission."""
from pathlib import Path
import unittest
import hashlib

from tests.c2.test_handler_order import parse
from tests.g3r.validate_trained_c2_eos_reference import verify


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/g3r/trained_c2_eos_reference.esk"


class TrainedC2EosReferenceSource(unittest.TestCase):
    def test_one_registry_and_exact_checkpoint(self):
        source = FIXTURE.read_text()
        self.assertTrue(parse(source))
        self.assertEqual(source.count('(load "g3r_trained_pair_producer_root.esk")'), 1)
        self.assertNotIn('g3c4-model-create-seeded-internal', source)
        self.assertNotIn('g3t-checked-m3t-initializer-state', source)
        ordered = (
            '(g3r-trained-pair-file-image (argument 3) 16777216)',
            '(cli3-b-restore! trainer (argument 3))',
            '(trainer-release! trainer)',
            '(module-eval! model)',
            '(g3m-generate-p1-g1! first-generator first-input)',
            '(g3r-committed-save-snapshot!',
            '(g3m-generate-p1-g1! eos-generator eos-input)',
        )
        at = 0
        for item in ordered:
            at = source.index(item, at) + len(item)
        self.assertIn('trained-digest (c1-sha256 trained-image)', source)
        self.assertIn('same-bytes?\n             trained-image', source)

    def test_eos_comes_from_genuine_sample_and_comparison_is_exact(self):
        source = FIXTURE.read_text()
        self.assertIn('(id (bytevector-u8-ref (vector-ref first-snapshot 2) 1))', source)
        self.assertIn('(policy id)', source)
        self.assertIn('(same-bytes? first-text eos-text)', source)
        self.assertIn('(same-bytes? (vector-ref first-snapshot 2)', source)
        self.assertIn('(same-words? (vector-ref first-snapshot 1)', source)
        self.assertIn('(= (vector-ref first-snapshot 6) 0)', source)
        self.assertIn('(= (vector-ref eos-snapshot 6) 1)', source)
        self.assertIn('(display " terminal=1 generated=1 history=2 rng=")', source)
        self.assertIn('(bytevector-u8-ref trained-digest index)', source)

    def test_receipt_binds_exact_trained_image_and_rng(self):
        trained = b"exact trained C2 example"
        prefix = ("G3-R TRAINED EOS REFERENCE PASS c2_sha256="
                  f"{hashlib.sha256(trained).hexdigest()} "
                  "prompt=41 id=a7 terminal=1 generated=1 history=2 ")
        line = prefix + "rng=1,1729,1,0\n"
        self.assertEqual(verify(line, trained, 65), (167, (1, 1729, 1, 0)))
        for bad, image, prompt in (
            (line, trained + b"!", 65),
            (line, trained, 66),
            (prefix + "rng=1,1729,0,0\n", trained, 65),
            (line.replace("terminal=1", "terminal=0"), trained, 65),
            (line + "extra", trained, 65),
        ):
            with self.subTest(bad=bad, image=image, prompt=prompt):
                with self.assertRaises(ValueError):
                    verify(bad, image, prompt)


if __name__ == "__main__":
    unittest.main()
