"""Package/link source contract for trained-C2 first-token EOS; no runtime proof."""
import hashlib
from pathlib import Path
import unittest

from tests.c2.test_handler_order import parse
from tests.g3r.validate_trained_c2_eos_reference import verify


ROOT = Path(__file__).resolve().parents[2]


class TrainedC2EosReferenceSource(unittest.TestCase):
    def test_one_package_two_exact_request_shapes(self):
        root = (ROOT / "native/g3r_trained_pair_producer_root.esk").read_text()
        extension = (ROOT / "native/g3r_trained_c2_eos_reference_extension.esk").read_text()
        caller = (ROOT / "tests/g3r/trained_c2_eos_reference.esk").read_text()
        self.assertTrue(parse(root) and parse(caller))
        self.assertEqual(len(parse(extension)), 1)
        self.assertEqual(root.count('(load "cli3_generate_root.esk")'), 1)
        self.assertEqual(root.count('(load "g3r_trained_c2_eos_reference_extension.esk")'), 1)
        projector = '(vector-set! t2-wave2-policy-projector 0 c2-policy-t2-entry-internal)'
        self.assertEqual(root.count(projector), 1)
        self.assertLess(root.index(projector), root.index('(cond\n    ((= (vector-length arguments) 7)'))
        self.assertLess(root.index('((= (vector-length arguments) 7)'),
                        root.index('((= (vector-length arguments) 5)'))
        self.assertIn('(g3r-trained-pair-produce-internal!', root)
        self.assertIn('(g3r-trained-c2-eos-reference-internal!', root)
        self.assertIn('(cli3-parse-i64 (vector-ref arguments 6) "PROMPT" 255)', root)
        self.assertIn('(cli3-parse-i64 (vector-ref arguments 4) "PROMPT" 255)', root)
        self.assertEqual((ROOT / 'native/g3r_trained_pair_private_renames.txt').read_text().splitlines(),
                         ['g3r-trained-pair-private-produce '
                          'et_e1b_private_g3r_trained_pair_produce_cabi_v1'])
        renames = [line.split() for name in (
            'native/tr3_public_installed_private_renames.txt',
            'native/cli3_generate_private_renames.txt',
            'native/g3r_trained_pair_private_renames.txt')
            for line in (ROOT / name).read_text().splitlines()]
        self.assertEqual(len(renames), 155)
        self.assertEqual(len({row[0] for row in renames}), 155)
        self.assertEqual(len({row[1] for row in renames}), 155)
        bridge = (ROOT / 'native/g3r_trained_pair_package_bridge.c').read_text()
        self.assertIn('et_e1b_private_g3r_trained_pair_produce_cabi_v1(', bridge)
        self.assertIn('et_g3r_private_test_trained_pair_produce_v1', caller)
        self.assertNotIn('(load "g3r_trained_pair_producer_root.esk")', caller)
        self.assertIn('(vector (vector (argument 0) (argument 1) (argument 2)', caller)

    def test_trained_restore_first_sample_eos_and_detached_result(self):
        source = (ROOT / 'native/g3r_trained_c2_eos_reference_extension.esk').read_text()
        ordered = (
            '(tokenizer (cli3-b-tokenizer tokenizer-path resolved))',
            '(g3r-trained-pair-file-image trained-c2 16777216)',
            '(cli3-b-restore! trainer trained-c2)',
            '(trainer-release! trainer)',
            '(module-eval! model)',
            '(g3m-generate-p1-g1! first-generator first-input)',
            '(g3r-committed-save-snapshot!',
            '(g3t-output-release! first-output)',
            '(g3t-generation-tensor-release! first-input)',
            '(g3t-generator-close! first-generator)',
            '(set! eos-generator (g3t-generator-create model tokenizer (policy id)))',
            '(g3m-generate-p1-g1! eos-generator eos-input)',
            '(set! answer',
            '(cleanup)',
        )
        at = 0
        for item in ordered:
            at = source.index(item, at) + len(item)
        self.assertIn('(id (bytevector-u8-ref (vector-ref first-snapshot 2) 1))', source)
        self.assertIn('(same-bytes? first-text eos-text)', source)
        self.assertIn('(same-words? (vector-ref first-snapshot 1)', source)
        self.assertIn('(= (vector-ref first-snapshot 6) 0)', source)
        self.assertIn('(= (vector-ref eos-snapshot 6) 1)', source)
        self.assertIn('(vector trained-digest prompt id (vector-ref eos-snapshot 1))', source)
        self.assertNotIn('g3c4-model-create-seeded-internal', source)

    def test_exact_root_closure(self):
        closure = (ROOT / 'native/g3r_trained_pair_producer_source_closure.txt').read_text().splitlines()
        self.assertEqual(len(closure), len(set(closure)))
        self.assertEqual(closure[0], 'native/g3r_trained_pair_producer_root.esk')
        self.assertEqual(closure[-1], 'native/g3r_trained_c2_eos_reference_extension.esk')
        self.assertTrue(all((ROOT / item).is_file() for item in closure))

    def test_receipt_binds_exact_trained_image_and_rng(self):
        trained = b'exact trained C2 example'
        prefix = ('G3-R TRAINED EOS REFERENCE PASS c2_sha256='
                  f'{hashlib.sha256(trained).hexdigest()} '
                  'prompt=41 id=a7 terminal=1 generated=1 history=2 ')
        line = prefix + 'rng=1,1729,1,0\n'
        self.assertEqual(verify(line, trained, 65), (167, (1, 1729, 1, 0)))
        for bad, image, prompt in (
            (line, trained + b'!', 65),
            (line, trained, 66),
            (prefix + 'rng=1,1729,0,0\n', trained, 65),
            (line.replace('terminal=1', 'terminal=0'), trained, 65),
            (line + 'extra', trained, 65),
        ):
            with self.subTest(bad=bad, image=image, prompt=prompt):
                with self.assertRaises(ValueError):
                    verify(bad, image, prompt)


if __name__ == '__main__':
    unittest.main()
