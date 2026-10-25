import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from tests.cpu4_c4_p3_fresh.contract import (BRIDGES, FEATURES, OWNER, T1_FP,
    units, validate_wire, validate_resumed)
from tests.cpu4_v1_aggregate.closure import ROOT, load_roster
from tests.cpu4_c4_p3_fresh.driver import (PRIOR_RECIPE_SHA, PRIOR_SOURCE,
    outer_receipt, prior_receipt)


def sample_wire():
    fields = bytearray(192)
    struct.pack_into('<IHH8IQ', fields, 0, 224, 1, 0, 0, 3, 4, 2, 1, 0, 0, 0, 2)
    struct.pack_into('<4q', fields, 48, 3, 197, 3, 0)
    struct.pack_into('<4I', fields, 80, 1, 1, 2, 0)
    struct.pack_into('<6q', fields, 96, 0, 0x3f800000, 256, 0x3f800000, 2, 41)
    struct.pack_into('<4q', fields, 144, 1, 41, 0, 0)
    struct.pack_into('<4I', fields, 176, 14, 4768, 1, 0)
    values, sha = bytes(4768), bytes(range(32))
    record = bytes(fields) + hashlib.sha256(
        b'cpu4-c4-capture-v1\0' + fields + values).digest()
    prefix = b'C4GRCV1\0' + struct.pack('<II', 400, 1) + record + sha + T1_FP
    return prefix + hashlib.sha256(prefix).digest(), values, sha


class FreshP3Contract(unittest.TestCase):
    def test_only_one_owner_gets_all_c4_features(self):
        roster = load_roster()
        overlay = json.loads((ROOT / 'tests/cpu4_trained_g2/normal_native_overlay.json').read_text())
        selected = units(roster, overlay)
        self.assertEqual(len(selected), 39)
        self.assertEqual([unit['source'] for unit in selected[-2:]], list(BRIDGES))
        self.assertEqual([unit['source'] for unit in selected
            if all(feature in unit['defines'] for feature in FEATURES)], [OWNER])
        for unit in selected:
            if unit['source'] != OWNER:
                self.assertFalse(set(unit['defines']) & set(FEATURES))
        bad = json.loads(json.dumps(roster))
        bad['common_defines'].append(FEATURES[0])
        with self.assertRaises((ValueError, AssertionError)):
            units(bad, overlay)

    def test_fixed_wire_rejects_identity_digest_and_profile_changes(self):
        wire, values, sha = sample_wire()
        validate_wire(wire, values, sha)
        for offset in (0, 16 + 20, 16 + 48 + 8, 16 + 144, 208, 240, 272, 368):
            changed = bytearray(wire)
            changed[offset] ^= 1
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                validate_wire(bytes(changed), values, sha)
        with self.assertRaises(ValueError): validate_wire(wire, values[:-1], sha)
        with self.assertRaises(ValueError): validate_wire(wire, values, bytes(32))

    def test_result_and_source_route_are_closed(self):
        with self.assertRaises(ValueError): validate_resumed(
            b'CPU4-P3-RESUMED 41 1 3 3 41 3 41 1 41 0 0\n'
            b'CPU4-P3-FRESH-PASS resume 11\n', b'')
        source = (ROOT / 'tests/cpu4_c4_p3_fresh/runtime.esk').read_text()
        self.assertLess(source.index('cpu4-checkpoint-factory-create-internal!'),
                        source.index('cpu4-c4-p3-checked-file-issue-internal!'))
        self.assertLess(source.index('cpu4-c4-p3-checked-file-issue-internal!'),
                        source.index('cpu4-c4-p3-checked-file-handoff-internal!'))
        self.assertLess(source.index('cpu4-c4-p3-checked-file-handoff-internal!'),
                        source.index('g3c4-p2g2-terminal-resume-private!'))
        self.assertIn('cpu4-c4-p3-live-produce!', source)
        self.assertNotIn('g3g-public-generator-generate', source)
        producer = (ROOT / 'native/cpu4_c4_p3_live_producer_extension.esk').read_text()
        checked = (ROOT / 'native/cpu4_c4_p3_checked_file_extension.esk').read_text()
        terminal = (ROOT / 'native/g3c4_p2g2_terminal_commit_extension.esk').read_text()
        self.assertEqual(producer.count(':real et_cpu4_private_c4_capture_copy_record_eshkol_v1'), 1)
        self.assertEqual(producer.count(':real et_cpu4_private_c4_capture_copy_parameters_eshkol_v1'), 1)
        self.assertEqual(producer.count(':real et_cpu4_private_c4_p3_wire_encode_eshkol_v1'), 1)
        self.assertIn(':real et_cpu4_private_c4_p3_wire_decode_eshkol_v1', checked)
        self.assertIn(':real et_cpu4_private_c4_p3_resume_eshkol_v1', terminal)

    def test_prior_receipt_seal_and_artifact_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            combined = Path(directory) / 'combined'
            combined.mkdir()
            (combined / 'driver-program.exit').write_text('0\n')
            (combined / 'driver-custody.exit').write_text('0\n')
            (combined / 'source.before.stdout').write_text('\n'.join(PRIOR_SOURCE) + '\n')
            (combined / 'source.after.stdout').write_text('\n'.join(PRIOR_SOURCE) + '\n')
            pins = {'two-token.final-b250.1.c2': 'a' * 64,
                    'second-eos.run1.stdout': 'b' * 64}
            sums = combined / 'SHA256SUMS'
            sums.write_text(''.join(digest + '  ' + name + '\n'
                                    for name, digest in pins.items()))
            seal = hashlib.sha256(sums.read_bytes()).hexdigest()
            (combined / 'SHA256SUMS.sha256').write_text(seal + '  SHA256SUMS\n')
            prior_receipt(Path(directory), seal, pins)
            with self.assertRaises(ValueError):
                prior_receipt(Path(directory), '0' * 64, pins)
            with self.assertRaises(ValueError):
                prior_receipt(Path(directory), seal, {**pins,
                    'second-eos.run1.stdout': 'c' * 64})
            (combined / 'driver-custody.exit').write_text('74\n')
            with self.assertRaises(ValueError):
                prior_receipt(Path(directory), seal, pins)

    def test_outer_receipt_requires_all_rows_and_success(self):
        with tempfile.TemporaryDirectory() as directory:
            prior = Path(directory)
            for name, content in {
                **{name: '0\n' for name in
                   ('docker.exit', 'launcher.exit', 'inside-program.exit',
                    'inside-custody.exit', 'inside.exit', 'native.exit')},
                'state': 'complete\n', 'started.utc': '2026-10-24T00:00:00Z\n',
                'finished.utc': '2026-10-24T00:01:00Z\n',
                'recipe.expected-sha256': PRIOR_RECIPE_SHA + '\n',
                **{label + '.stderr': '' for label in
                   ('recipe-precheck', 'recipe-after-lock', 'preflight',
                    'recipe-postcheck', 'postflight')},
                'preflight.stdout': json.dumps({'preflight': 'PASS',
                    'source': PRIOR_SOURCE[0]}) + '\n',
                'postflight.stdout': json.dumps({'preflight': 'PASS',
                    'source': PRIOR_SOURCE[0]}) + '\n',
            }.items():
                (prior / name).write_text(content)
            sums = prior / 'SHA256SUMS-RUN'
            sums.write_text(''.join(
                hashlib.sha256(path.read_bytes()).hexdigest() + '  ./' +
                path.name + '\n' for path in sorted(prior.iterdir())))
            seal = hashlib.sha256(sums.read_bytes()).hexdigest()
            outer_receipt(prior, seal)
            with self.assertRaises(ValueError): outer_receipt(prior, '0' * 64)
            (prior / 'launcher.exit').write_text('74\n')
            with self.assertRaises(ValueError): outer_receipt(prior, seal)


if __name__ == '__main__': unittest.main()
