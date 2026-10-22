"""Cheap actual-plan/caller controls; does not execute any compiler or runtime."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import subprocess
from tests.cpu4_trainer_control.test_source_contract import parse, walk
from .file_driver import CALLER, OVERLAY, ROOT, OBSERVER, TEST_FLAGS, dependencies, gate, load_roster, plan


class FileDriver(unittest.TestCase):
    def test_one_actual_root_then_unchanged_fixtures_and_call(self):
        forms = parse(CALLER.read_text())
        self.assertEqual(forms[:3], [['load', 'cpu4_v1_private_root.esk'],
                                     ['load', 'nonregional_runtime.esk'], ['load', 'positive_runtime.esk']])
        self.assertEqual(len(forms), 4)
        self.assertEqual(forms[3][0], 'guard')
        self.assertEqual(forms[3][2:], [['cpu4-factory-data-runtime!', ['car', ['cdr', ['command-line']]],
                                      ['car', ['cdr', ['cdr', ['command-line']]]]]])
        for name, digest in {
            'nonregional_runtime.esk': 'fa3c7a8bd87512d109c95f00bb500fd8b8faf8f8a6a0f052ce89066c20e88cd7',
            'positive_runtime.esk': 'ce6fae7dc9fd8be4caee0602ab484b3ef8e72cc4ea50672200f76bad2c202a24',
        }.items():
            self.assertEqual(hashlib.sha256((CALLER.parent / name).read_bytes()).hexdigest(), digest)

    def test_unexpected_error_guard_reports_real_fields_and_exits(self):
        guarded = parse(CALLER.read_text())[3]
        self.assertEqual(guarded[0], 'guard')
        self.assertEqual(guarded[1][0], 'caught')
        handler = guarded[1][1]
        self.assertEqual(handler[0], '#t')
        self.assertEqual(handler[-1], ['exit', '1'])
        calls = [node[0] for node in walk(handler) if node and isinstance(node[0], str)]
        for accessor in ('transformer-error-category', 'transformer-error-operation',
                         'transformer-error-message', 'transformer-error-details'):
            self.assertEqual(calls.count(accessor), 1)
        self.assertIn(['write', 'caught'], list(walk(handler)))
        self.assertIn('CPU4 FILE unexpected FOREIGN value=', CALLER.read_text())
        self.assertIn('CPU4 FILE unexpected E1 category=', CALLER.read_text())
        self.assertIn(['display', 'cpu4-fdt-checks'], list(walk(handler)))
        self.assertIn(['display', 'cpu4-factory-region-checks'], list(walk(handler)))
        self.assertNotIn('let', calls)
        self.assertNotIn('cpu4-factory-data-runtime!', calls)
        # The success branch is only the original real call, without prints.
        self.assertEqual(len(guarded[2:]), 1)
        self.assertEqual(guarded[2][0], 'cpu4-factory-data-runtime!')

    def test_overlay_preserves_every_production_tu_feature_and_owner(self):
        roster = load_roster()
        before = copy.deepcopy(roster)
        selected = plan(roster, json.loads(OVERLAY.read_text()))
        self.assertEqual(roster, before)
        self.assertEqual(selected[-1], OBSERVER)
        self.assertEqual(len(selected), len(roster['translation_units']) + 1)
        for original, actual in zip(roster['translation_units'], selected):
            self.assertEqual(original['source'], actual['source'])
            self.assertEqual(actual['defines'], original['defines'] + TEST_FLAGS.get(original['source'], []))
        self.assertEqual(len({unit['source'] for unit in selected}), len(selected))

    def test_missing_hooks_duplicate_owner_and_sanitize_rejected(self):
        overlay = json.loads(OVERLAY.read_text())
        for source in TEST_FLAGS:
            changed = copy.deepcopy(overlay)
            del changed['per_tu_defines'][source]
            with self.subTest(source=source), self.assertRaises(AssertionError):
                plan(load_roster(), changed)
        changed = copy.deepcopy(overlay)
        changed['additional_translation_units'].append({'source': 'native/f32_tensor.c', 'defines': []})
        with self.assertRaises(AssertionError):
            plan(load_roster(), changed)
        changed = copy.deepcopy(overlay)
        changed['mode'] = 'sanitize'
        with self.assertRaises(AssertionError):
            plan(load_roster(), changed)
        roster = load_roster()
        roster['common_defines'].remove('ET_CPU4_CANDIDATE_PRIVATE')
        with self.assertRaises(AssertionError):
            plan(roster, overlay)

    def test_raw_failure_and_custody_failure_remain_distinct(self):
        for custody_failure in (False, True):
            with self.subTest(custody_failure=custody_failure), tempfile.TemporaryDirectory() as temporary:
                out = Path(temporary) / 'fresh'
                def git_reply(argv, **kwargs):
                    return b'commit\ntree\n' if 'rev-parse' in argv else b''
                def reject_native(argv, **kwargs):
                    return subprocess.CompletedProcess(argv, 37)
                with patch('tests.cpu4_checkpoint_factory.file_driver.subprocess.check_output', side_effect=git_reply), \
                     patch('tests.cpu4_checkpoint_factory.file_driver.subprocess.run', side_effect=reject_native):
                    if custody_failure:
                        with patch('tests.cpu4_checkpoint_factory.file_driver.inventory_depfiles', side_effect=ValueError('changed dependency')):
                            status = gate(out, Path('/pinned/runner'), Path('/pinned/cc'), Path('/pinned/cxx'), Path('/pinned/source'))
                    else:
                        status = gate(out, Path('/pinned/runner'), Path('/pinned/cc'), Path('/pinned/cxx'), Path('/pinned/source'))
                self.assertEqual(status, 74 if custody_failure else 37)
                self.assertEqual((out / 'driver-program.exit').read_text(), '37\n')
                self.assertEqual((out / 'driver-custody.exit').read_text(), '74\n' if custody_failure else '0\n')
                for line in (out / 'SHA256SUMS').read_text().splitlines():
                    digest, name = line.split('  ', 1)
                    self.assertEqual(hashlib.sha256((out / name).read_bytes()).hexdigest(), digest)

    def test_discovery_parser_preserves_real_escaped_depfile_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            depfile = directory / 'actual.d'
            depfile.write_text('out.o: native/a.c \\\n native/a.h path\\ with\\ spaces/header.h\n')
            self.assertEqual(dependencies(depfile), [(ROOT / name).resolve() for name in
                ('native/a.c', 'native/a.h', 'path with spaces/header.h')])


if __name__ == '__main__':
    unittest.main()
