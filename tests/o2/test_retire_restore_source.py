"""Cheap causal source guards only; no compiler/native/runtime admission."""
from pathlib import Path
import re
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]
BASE = '2740d4c98c56ee4de41864fb9d8cacc10710d5e9'


def function(text, name):
    start = text.index(name + '(')
    start = text.index('{', start)
    depth = 1
    end = start + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


def off(text):
    """Remove only this feature's complete blocks; preserve other directives."""
    result, stack = [], []
    for line in text.splitlines():
        if re.match(r'\s*#\s*(if|ifdef|ifndef)\b', line):
            stack.append('ifdef ET_G3R_CANDIDATE_RETIRE_PRIVATE' in line or any(stack))
            if stack[-1]:
                continue
        elif re.match(r'\s*#\s*endif\b', line):
            removed = stack.pop()
            if removed:
                continue
        if not any(stack):
            result.append(line)
    return ''.join(''.join(result).split())


def validate(o2, tr3):
    helper = function(o2, 'et_o2_retire_record_matches')
    for token in ('record != et_o2_retire_record_find(owner)', 'record->owner != owner',
                  'record->live != 1u', 'owner->entries != record->entries',
                  'owner->count != record->count', 'sizeof(et_o2_entry)'):
        assert token in helper, token
    load = function(o2, 'et_o2_optimizer_load_state_v1')
    assert load.count('et_o2_retire_record_matches(optimizer, retire_record)') == 2
    assert load.rindex('et_o2_retire_record_matches') < load.index('et_f32_tensor_copy_plan_commit_v1')
    assert load.count('retire_record->roster[index].options = state->entries[index].options;') == 1
    commit = function(tr3, 'et_o2_tr3_c_restore_commit_native_v1')
    assert commit.count('plan->retire_record->roster[index].options = plan->target_options[index];') == 1
    assert 'memcpy' not in commit and 'record_find' not in commit and 'record_matches' not in commit
    sealed = function(tr3, 'et_o2_tr3_c_restore_sealed_state')
    assert 'record_find' not in sealed and 'record_matches' not in sealed
    assert 'memcmp(entry, &plan->retire_record->roster[index], sizeof(*entry))' in sealed
    for name in ('et_o2_tr3_c_restore_receiver', 'et_o2_tr3_c_restore_baseline'):
        assert 'et_o2_retire_record_matches' in function(tr3, name)
    assert tr3.count('plan->retire_record = et_o2_retire_record_find(optimizer);') == 1
    assert 'plan->retire_record = NULL;' in function(tr3, 'et_o2_tr3_c_restore_clear_pointers')


class Source(unittest.TestCase):
    def setUp(self):
        self.o2 = (ROOT / 'native/o2_optimizer.c').read_text()
        self.tr3 = (ROOT / 'native/tr3_c_o2_restore.inc').read_text()

    def test_contract(self):
        validate(self.o2, self.tr3)

    def test_feature_off_original_forms(self):
        for path in ('native/o2_optimizer.c', 'native/tr3_c_o2_restore.inc'):
            original = subprocess.check_output(['git', 'show', BASE + ':' + path], cwd=ROOT, text=True)
            self.assertEqual(off(original), off((ROOT / path).read_text()))

    def test_causal_mutations(self):
        for o2, tr3 in (
            (self.o2.replace('record != et_o2_retire_record_find(owner)', '0', 1), self.tr3),
            (self.o2.replace('retire_record->roster[index].options = state->entries[index].options;', '', 1), self.tr3),
            (self.o2, self.tr3.replace('plan->retire_record->roster[index].options = plan->target_options[index];', '', 1)),
            (self.o2, self.tr3.replace('memcmp(entry, &plan->retire_record->roster[index], sizeof(*entry))', '0', 1)),
            (self.o2, self.tr3.replace('plan->retire_record->roster[index].options = plan->target_options[index];', 'memcpy(plan->retire_record->roster, plan->owner->entries, 14 * sizeof(et_o2_entry));', 1)),
        ):
            with self.assertRaises(AssertionError):
                validate(o2, tr3)

    def test_actual_probes_and_resource_contract(self):
        ordinary = (ROOT / 'tests/o2/test_g3r_o2_retire.c').read_text()
        native = (ROOT / 'tests/o2/test_tr3_c_o2_restore.c').read_text()
        header = (ROOT / 'tests/o2/retire_restore_probe.h').read_text()
        for call in ('reconstruct_create', 'reconstruct_set', 'reconstruct_prepare', 'reconstruct_commit'):
            self.assertIn('et_o2_optimizer_state_' + call + '_v1', ordinary)
        self.assertIn('et_o2_optimizer_load_state_v1(f.optimizer, state, &error) == 0', ordinary)
        self.assertNotIn('test_state_set_option', ordinary)
        for call in ('et_tr3_c_i2_copy_builder_prepare_restore42_v1',
                     'et_tr3_c_i2_copy_builder_commit_restore42_checked_v1',
                     'et_o2_tr3_c_restore_commit_native_v1', 'et_o2_tr3_c_restore_finalize_v1'):
            self.assertIn(call, function(native, 'probe_tr3_restore'))
        self.assertIn('et_o2_private_candidate_retire_preflight_v1(optimizer, NULL)', header)
        self.assertIn('et_o2_private_candidate_retire_preflight_v1(optimizer, &error)', header)
        self.assertIn('optimizer entry differs from its published roster', header)
        self.assertIn('retained_delta == 2u * sizeof(*plan)', native)
        self.assertIn('sizeof(*plan) == 1472u', native)
        self.assertIn('sizeof(*plan) == 1464u', native)
        self.assertIn('state->active_borrows == 0u', header)
        self.assertIn('field < 11u', ordinary)
        self.assertIn('field < 11u', function(native, 'probe_tr3_restore'))


if __name__ == '__main__':
    unittest.main()
