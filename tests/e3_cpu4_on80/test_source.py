"""Cheap projection/tuple/negative checks; no compiled admission."""
import ast
import copy
import hashlib
import subprocess
import unittest
from pathlib import Path
from tests.cpu4_v1_aggregate.closure import ROOT, load_roster, validate
from tests.e3_cpu4_on80.source_contract import validate_text, validate_variant
from tests.e3_cpu4_on80.normal_driver import preflight, DEPENDENCY_PATHS, HELPER
from tests.e3_cpu4_on80.check_output import check

ORIGINAL = '4c745a0fad134cab1a57d29e80157be9b2b39a0b'


def transcript(checks):
    return ('CPU4-CONSUMER-REPORT #(1085385783 1090519040 1132521808 0 8 2)\n'
            f'CPU4-CONSUMER-PASS checks={checks}\n')


class SourceTests(unittest.TestCase):
    def test_original_fixture_helper_resources_untouched(self):
        for path in ('tests/e3_cpu4_consumer/runtime.esk', HELPER,
                     'tests/e3_cpu4_consumer/resources.py', 'tests/e3_cpu4_consumer/observation.py'):
            self.assertEqual((ROOT / path).read_bytes(), subprocess.check_output(
                ['git', 'show', ORIGINAL + ':' + path], cwd=ROOT))

    def test_exact_projection_and_seed_lifecycle_contract(self):
        self.assertTrue(validate_variant(ROOT))

    def test_projection_rejects_each_lifecycle_or_assertion_change(self):
        original = (ROOT / 'tests/e3_cpu4_consumer/runtime.esk').read_text()
        current = (ROOT / 'tests/e3_cpu4_on80/runtime.esk').read_text()
        mutants = [current.replace('cpu4_v1_private_root.esk', 'e3_cpu4_private_root.esk'),
                   current.replace('(cpu4-candidate-model-create-internal! resolved)',
                                   '(g3c4-model-create-seeded-internal 1729)'),
                   current.replace('(cpu4-candidate-model-internal candidate)', 'candidate'),
                   current.replace('(cpu4-candidate-retire-internal! candidate)', '#t'),
                   current.replace('(cpu4-candidate-retire-internal! candidate)', '')
                          .replace('(module-train! model)', '(cpu4-candidate-retire-internal! candidate)\n(module-train! model)'),
                   current.replace("(check 'late-denial-keeps-prior", "(check 'late-denial-relabelled"),
                   current.replace('run.seed\\\":1729', 'run.seed\\\":1730'),
                   current.replace('(module-train! model)', '(module-eval! model)')]
        for mutant in mutants:
            with self.subTest(change=mutants.index(mutant)):
                self.assertNotEqual(mutant, current)
                with self.assertRaises(AssertionError):
                    validate_text(original, mutant)

    def test_real_35_roster_plus_original_helper_only(self):
        roster = load_roster()
        units = preflight(ROOT, roster)
        self.assertEqual(len(units), 36)
        self.assertEqual(units[:-1], roster['translation_units'])
        self.assertEqual(units[-1], {'source': HELPER, 'defines': []})
        self.assertEqual(len({unit['source'] for unit in units}), 36)
        self.assertFalse(any('TESTING' in flag for unit in units for flag in unit['defines']))

    def test_candidate_off_duplicate_owner_m3t_mutations_reject(self):
        roster = load_roster()
        mutants = []
        off = copy.deepcopy(roster); off['candidate_mode'] = 'off'; mutants.append(off)
        off_flag = copy.deepcopy(roster); off_flag['common_defines'].remove('ET_CPU4_CANDIDATE_PRIVATE'); mutants.append(off_flag)
        duplicate = copy.deepcopy(roster); duplicate['translation_units'].append(duplicate['translation_units'][0]); mutants.append(duplicate)
        m3t = copy.deepcopy(roster); m3t['common_defines'].append('ET_M3T_PACKAGE_BUILD'); mutants.append(m3t)
        for mutant in mutants:
            with self.assertRaises(AssertionError):
                validate(ROOT, mutant)

    def test_exact_expected_words_and_totals(self):
        self.assertEqual(check(transcript(217), 'absent')['checks'], 217)
        self.assertEqual(check(transcript(219), 'present')['checks'], 219)
        for mode, text in [('absent', transcript(219)), ('present', transcript(217)),
                           ('absent', transcript(217).replace('1085385783', '1085385784')),
                           ('present', transcript(219) + 'EXTRA\n'),
                           ('other', transcript(217))]:
            with self.assertRaises(ValueError):
                check(text, mode)

    def test_driver_closed_two_processes_and_portable_helpers(self):
        source = (ROOT / 'tests/e3_cpu4_on80/normal_driver.py').read_text()
        current = ast.parse(source, feature_version=(3, 10))
        baseline = ast.parse((ROOT / 'tests/cpu4_trained_g2/normal_driver.py').read_text())
        for name in ('hashes', 'dependencies', 'inventory_depfiles'):
            def function(tree):
                return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
            self.assertEqual(ast.dump(function(current)), ast.dump(function(baseline)))
        self.assertIn("MODES = ('absent', 'present')", source)
        self.assertNotIn('for repeat in', source)
        self.assertNotIn('-fsanitize', source)
        self.assertIn("'1200s', runner", source)
        self.assertIn("'120s',\n                        executable, corpus, mode", source)
        self.assertNotIn('file_digest', source)
        self.assertEqual(set(DEPENDENCY_PATHS), {row.split('  ', 1)[1] for row in
            (ROOT / 'tests/e3_cpu4_on80/DEPENDENCIES.sha256').read_text().splitlines()})


if __name__ == '__main__':
    unittest.main()
