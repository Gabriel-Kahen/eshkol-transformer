import hashlib
from pathlib import Path
import tempfile
import unittest

from tests.cpu4_b250_fresh.witness import SIZES, compare, planes, seal
from tests.cpu4_trainer_control.test_source_contract import parse

ROOT = Path(__file__).resolve().parents[2]


class FreshB250Source(unittest.TestCase):
    def test_producer_before_save(self):
        text = (ROOT / 'tests/cpu4_trained_g2/generation.esk').read_text()
        parse(text)
        assert text.index('(cpu4-trained-g2-emit-live-b250! final-b)') < text.index('(c2-public-checkpoint-save!')
        assert text.index('CPU4-LIVE-B250-PLANE') < text.index('(define (cpu4-trained-g2-save-b250!')
        assert '(* 4 (cpu4-checkpoint-product' in text

    def test_fresh_factory_and_handoff_authority(self):
        text = (ROOT / 'tests/cpu4_b250_fresh/runtime.esk').read_text()
        parse(text)
        assert text.index('(cpu4-checkpoint-factory-create-internal!') < text.index('(cpu4-resume-snapshot-copy!')
        assert text.index('(cpu4-resume-snapshot-copy!') < text.index('(cpu4-checkpoint-factory-generator-handoff-internal!')
        assert text.index('(cpu4-resume-snapshot-copy!') < text.rindex('(cpu4-b250-fresh-emit! copy)')
        for required in ('cpu4-resume-cleanup! ledger', 't1-wave1-tokenizer-fingerprint',
                         'cpu4-resume-counts! trainer 250', 'g3c4-public-g2-profile-live',
                         'g3g-public-generator-close', 'CPU4-B250-FRESH-PENDING'):
            assert required in text

    def test_receipt_parser_and_mutations(self):
        rows = [f'CPU4-LIVE-B250-PLANE {i} {size} ' + ' '.join('0' for _ in range(size))
                for i, size in enumerate(SIZES)]
        stdout = ('\n'.join(rows) + '\n').encode()
        self.assertEqual(b''.join(planes(stdout, 'CPU4-LIVE-B250-PLANE')), bytes(4768))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sealed'
            meta = seal(stdout, 'CPU4-LIVE-B250-PLANE', path)
            self.assertEqual(meta['values_sha256'], hashlib.sha256(bytes(4768)).hexdigest())
            compare(path, stdout.replace(b'LIVE', b'FRESH'), 'CPU4-FRESH-B250-PLANE')
            for mutant in (stdout + rows[0].encode() + b'\n',
                           stdout.replace(b'PLANE 0 64', b'PLANE 0 65', 1),
                           stdout.replace(b'PLANE 0 64 0', b'PLANE 0 64 256', 1),
                           stdout.replace(b'PLANE 0 64 0', b'PLANE 0 64 00', 1)):
                with self.assertRaises(ValueError):
                    planes(mutant, 'CPU4-LIVE-B250-PLANE')
            wrong = stdout.replace(b'PLANE 0 64 0', b'PLANE 0 64 1', 1)
            with self.assertRaises(ValueError):
                compare(path, wrong.replace(b'LIVE', b'FRESH'), 'CPU4-FRESH-B250-PLANE')


if __name__ == '__main__':
    unittest.main()
