"""Cheap definition/ownership/sequence guards; no native or Eshkol execution."""
import hashlib
import json
import re
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests/cpu4_trainer_control'))
from test_source_contract import parse, walk

BODY = ROOT / 'tests/cpu4_resume/body.esk'


def calls(form, name):
    return [n for n in walk(form) if n and n[0] == name]


class Source(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = BODY.read_text()
        cls.forms = parse(cls.text)
        cls.functions = {f[1][0]: f for f in cls.forms if f[0] == 'define' and isinstance(f[1], list)}
        cls.main = cls.functions['cpu4-heldout-resume-runtime!']

    def test_definitions_only_and_exact_existing_dependencies(self):
        self.assertTrue(all(f[0] == 'define' for f in self.forms))
        for line in (ROOT / 'tests/cpu4_resume/DEPENDENCIES.sha256').read_text().splitlines():
            digest, path = line.split('  ')
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest, path)
            pinned = subprocess.check_output(['git', 'show', '4d2c05cc:'+path], cwd=ROOT)
            self.assertEqual(hashlib.sha256(pinned).hexdigest(), digest)
        own = set(self.functions)
        include_dirs = json.loads((ROOT / 'native/cpu4_v1_private_inputs.json').read_text())['source_include_dirs']
        seen = set()
        def source_closure(path):
            if path in seen:
                return ''
            seen.add(path)
            text = path.read_text()
            for form in parse(text):
                if form[0] == 'load':
                    child = next((ROOT / directory / form[1] for directory in include_dirs
                                  if (ROOT / directory / form[1]).is_file()), None)
                    self.assertIsNotNone(child, form[1])
                    text += '\n' + source_closure(child)
            return text
        dependencies = source_closure(ROOT / 'native/cpu4_v1_private_root.esk')
        dependencies += (ROOT / 'tests/cpu4_heldout/body.esk').read_text()
        for f in self.forms:
            for n in walk(f):
                if n and isinstance(n[0], str) and n[0].startswith(('cpu4-', 'c2-', 'heldout-')) and n[0] not in own:
                    self.assertTrue(re.search(r'\(define(?:-syntax)?\s+\(?' + re.escape(n[0]) + r'(?=[\s)])', dependencies)
                                    or re.search(r'\(extern\s+\w+\s+' + re.escape(n[0]) + r'(?=[\s)])', dependencies), n[0])
        for forbidden in ('cpu4-fdt-', 'seed-p1', 'seed-o2', 'restore-entries!',
                          'state-dict-adopt-owned', 'o2-state-reconstruct', 'tr3-lease-',
                          'm3t-public-model-create', 'g3g-', 'set! heldout-fixture-session'):
            self.assertNotIn(forbidden, self.text)
        self.assertFalse(calls(self.main, 'cpu4-trainer-restore-internal!'))

    def test_exact_optimizer_and_actual_375_step_sequence(self):
        config = self.functions['cpu4-resume-config']
        self.assertIn(['quote', ['none']], list(walk(config)))
        self.assertIn(['quote', ['constant']], list(walk(config)))
        for bit in ('3dcccccd', '3f000000', '3a83126f', '00000000'):
            self.assertIn('"' + bit + '"', self.text)
        self.assertEqual(calls(self.main, 'cpu4-resume-run!'), [
            ['cpu4-resume-run!', 'a', '1', '125'],
            ['cpu4-resume-run!', 'a', '126', '250'],
            ['cpu4-resume-run!', 'b', '126', '250']])
        self.assertEqual(calls(self.main, 'cpu4-heldout-evaluate-internal!'), [
            ['cpu4-heldout-evaluate-internal!', 'a', 'heldout', ['quote', 'baseline']],
            ['cpu4-heldout-evaluate-internal!', 'a', 'heldout', ['quote', 'post']]])
        text = self.text[self.text.index('(define (cpu4-heldout-resume-runtime!'):]
        self.assertLess(text.index('(c2-public-checkpoint-save!'), text.index('(cpu4-checkpoint-factory-create-internal!'))
        self.assertLess(text.index('(c2-training-state-release-internal!'), text.index('(cpu4-checkpoint-factory-create-internal!'))
        self.assertLess(text.index('(cpu4-resume-counts! b 125)'), text.index('(cpu4-resume-run! b'))
        self.assertLess(text.index('(cpu4-resume-copy=? at125 restored)'), text.index('(cpu4-resume-run! b'))
        self.assertIn('(= cpu4-resume-steps 125)', text)
        self.assertIn('(= cpu4-resume-steps 250)', text)
        self.assertIn('(= cpu4-resume-steps 375)', text)
        self.assertIn('(cpu4-resume-snapshot-copy! a (vector-ref ledger 2))', text)
        self.assertIn('(not (cpu4-resume-values=? (vector-ref at0 0) (vector-ref at125 0)))', text)

    def test_full_capture_and_cross_owner_comparison(self):
        capture = self.functions['cpu4-resume-capture']
        for name in ('c2-training-state-borrow-begin-internal', 'c2-training-state-borrow-projection-internal',
                     'state-dict-c2-owned-tensor-borrow-begin-internal', 'c2-model-checkpoint-encode',
                     'o2-state-copy-moment-bits-c2-owned-internal', 'cpu4-checkpoint-controls-copy',
                     'o2-copy-config-data', 'o2-copy-paths', 'o2-copy-aliases'):
            self.assertTrue(calls(capture, name), name)
        comparator = self.functions['cpu4-resume-copy=?']
        self.assertEqual(len(calls(comparator, 'd2-core-bytes=?')), 1)
        self.assertTrue(calls(comparator, 'cpu4-resume-values=?'))
        self.assertIn(['=', 'index', '14'], list(walk(self.functions['cpu4-resume-values=?'])))
        self.assertIn(['=', 'index', '28'], list(walk(comparator)))
        self.assertTrue(calls(comparator, 'heldout-controls=?'))
        for field in ('3', '4'):
            self.assertIn(['equal?', ['vector-ref', 'expected', field], ['vector-ref', 'actual', field]], list(walk(comparator)))
        cleanup = self.functions['cpu4-resume-capture-cleanup!']
        heads = [n[0] for n in walk(cleanup) if n and isinstance(n[0], str)]
        self.assertLess(heads.index('state-dict-c2-owned-tensor-borrow-end-internal'),
                        heads.index('c2-training-state-borrow-end-internal'))

    def test_fresh_B_e3_full_preservation_without_one_shot_ledger_mutation(self):
        evaluator = self.functions['cpu4-resume-evaluate-b!']
        for name in ('cpu4-e3-frame-bind-internal', 'e3-evaluate-into!', 'e3-counter-ref', 'e3-frame-unbind!',
                     'heldout-copy-controls!', 'heldout-modes', 'heldout-image-create',
                     'heldout-image-equal', 'heldout-cleanup!'):
            self.assertTrue(calls(evaluator, name), name)
        self.assertNotIn('heldout-fixture-session', self.text)
        self.assertIn(['equal?', 'post-a', 'post-b'], list(walk(self.main)))
        text = self.text[self.text.index('(define (cpu4-resume-evaluate-b!'):]
        text = text[:text.index('(define (cpu4-resume-cleanup!')]
        self.assertLess(text.index('(e3-frame-unbind!'), text.index('(heldout-controls=?'))
        cleanup = self.functions['cpu4-resume-cleanup!']
        heads = [n[0] for n in walk(cleanup) if n and isinstance(n[0], str)]
        self.assertLess(heads.index('c2-training-state-release-internal!'),
                        heads.index('cpu4-checkpoint-factory-retire-internal!'))
        self.assertLess(heads.index('cpu4-checkpoint-factory-retire-internal!'),
                        heads.index('cpu4-owned-candidate-retire-internal!'))

    def test_named_loop_guard_boundary_and_nonbuiltin_names(self):
        for f in self.forms:
            for n in walk(f):
                if n and n[0] == 'if':
                    self.assertIn(len(n), (3, 4))
                if len(n) > 2 and n[0] == 'let' and isinstance(n[1], str):
                    self.assertNotIn(n[1], ('values', 'match', 'length', 'vector', 'list', 'map', 'train'))
                    self.assertFalse(calls(n, 'guard'))
        self.assertEqual(len(calls(self.functions['cpu4-resume-step!'], 'cpu4-training-step-internal!')), 1)
        self.assertFalse(calls(self.functions['cpu4-resume-run!'], 'cpu4-training-step-internal!'))

    def test_exact_successful_check_budget(self):
        def immediate(form):
            return len(calls(form, 'cpu4-resume-check'))
        self.assertEqual(immediate(self.functions['cpu4-resume-counts!']), 1)
        self.assertEqual(immediate(self.functions['cpu4-resume-stream!']), 1)
        run = self.functions['cpu4-resume-run!']
        per_step = immediate(run) + len(calls(run, 'cpu4-resume-counts!'))
        self.assertEqual(per_step, 2)
        evaluate = self.functions['cpu4-resume-evaluate-b!']
        evaluation = immediate(evaluate) + len(calls(evaluate, 'cpu4-resume-counts!'))
        budget = immediate(self.main) + len(calls(self.main, 'cpu4-resume-stream!'))
        budget += len(calls(self.main, 'cpu4-resume-counts!')) + evaluation
        budget += sum((int(call[3])-int(call[2])+1)*per_step for call in calls(self.main, 'cpu4-resume-run!'))
        self.assertEqual(budget, 770)


if __name__ == '__main__':
    unittest.main()
