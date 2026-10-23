"""Constructor publication order and bounded cleanup contract; source only."""
from pathlib import Path
import subprocess
import unittest
from tests.cpu4_trainer_control.test_source_contract import parse, walk

ROOT = Path(__file__).resolve().parents[2]
PATH = 'native/cpu4_model_binding_extension.esk'
BASE = '79ea814d994417d646835f5d09f26459869bf4a7'


def constructor(forms):
    return next(f for f in forms if f[:2] == ['define', ['cpu4-model-binding-create-internal', 'model', 'resolved']])


def contract(form):
    guards = [n for n in walk(form) if n and n[0] == 'guard']
    assert len(guards) == 1
    guarded = guards[0]
    handler = guarded[1][1]
    assert guarded[2] == ['vector-set!', 'cpu4-model-bindings', '0', 'next'], 'publication must be guarded first body operation'
    canonical = guarded[3]
    assert canonical[:2] == ['let', [['canonical', ['car', ['vector-ref', 'cpu4-model-bindings', '0']]]]]
    nodes = list(walk(handler))
    assert ['vector-ref', 'cpu4-model-bindings', '0'] in nodes, 'handler must reread actual registry'
    assert ['eq?', ['vector-ref', 'pending', '4'], 'owner'] in nodes
    assert ['eq?', ['vector-ref', 'pending', '1'], ['quote', 'pending']] in nodes
    assert ['vector-set!', 'pending', '1', ['quote', 'dead']] in nodes
    assert handler[-1] == ['raise', 'caught'], 'preserve exact original caught value'
    for n in nodes:
        assert not any(item == 'canonical' or item == 'record' or item == 'next' for item in n), 'no stale authority capture'
    publications = [n for n in walk(form) if n and n[:3] == ['vector-set!', 'cpu4-model-bindings', '0']]
    assert publications == [guarded[2]]
    return guarded


class GuardSource(unittest.TestCase):
    def test_repaired_order_and_exact_native_tail(self):
        current = constructor(parse((ROOT/PATH).read_text()))
        guarded = contract(current)
        body = guarded[3][2:]
        self.assertEqual([node[0] for node in body], ['cpu4-binding-p1-check', 'cpu4-binding-roster!', 'let'])
        tail = body[-1]
        self.assertEqual(tail[1], [['lease', ['cpu4-native-lease-create', 'owner']]])
        self.assertEqual(tail[-3:], [['vector-set!', 'canonical', '5', 'lease'],
                                    ['vector-set!', 'canonical', '1', ['quote', 'live']],
                                    ['vector-ref', 'canonical', '0']])

    def test_original_order_fails_and_all_other_source_forms_unchanged(self):
        before = parse(subprocess.check_output(['git','show',BASE+':'+PATH],cwd=ROOT,text=True))
        after = parse((ROOT/PATH).read_text())
        with self.assertRaisesRegex(AssertionError, 'publication must be guarded'):
            contract(constructor(before))
        self.assertEqual([f for f in before if f != constructor(before)],
                         [f for f in after if f != constructor(after)])
        # The original checked P1/roster/native-create/store tail is identical.
        original_guard = next(n for n in walk(constructor(before)) if n and n[0] == 'guard')
        self.assertEqual(original_guard[2:], contract(constructor(after))[3][2:])

    def test_handler_rejects_unrelated_state_or_stale_record_mutations(self):
        import copy
        good = constructor(parse((ROOT/PATH).read_text()))
        for kind in ('owner', 'pending', 'stale', 'rethrow'):
            bad = copy.deepcopy(good)
            guard = next(n for n in walk(bad) if n and n[0] == 'guard')
            nodes = list(walk(guard[1][1]))
            if kind == 'owner':
                next(n for n in nodes if n == ['eq?', ['vector-ref','pending','4'],'owner'])[-1] = 'other-owner'
            elif kind == 'pending':
                next(n for n in nodes if n == ['eq?', ['vector-ref','pending','1'],['quote','pending']])[-1] = ['quote','live']
            elif kind == 'stale':
                next(n for n in nodes if n == ['vector-set!','pending','1',['quote','dead']])[1] = 'canonical'
            else:
                guard[1][1][-1] = ['raise',['vector','caught']]
            with self.subTest(kind=kind), self.assertRaises(AssertionError):
                contract(bad)


if __name__ == '__main__':
    unittest.main()
