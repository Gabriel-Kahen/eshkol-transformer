"""Mutation-sensitive ownership order for the uncompiled private scope."""
from pathlib import Path
import unittest

from tests.c2.test_handler_order import form, parse


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "native/g3r_same_owner_scope_extension.esk"
LOAD_ROOT = ROOT / "native/g3r_same_owner_scope_root.esk"


def define(forms, name):
    return next(x for x in forms if x[0] == "define" and
                isinstance(x[1], list) and x[1][0] == name)


def contains(tree, target):
    return tree == target or (isinstance(tree, list) and
                              any(contains(x, target) for x in tree))


def calls(tree):
    if not isinstance(tree, list):
        return []
    return ([tree[0]] if tree and isinstance(tree[0], str) else []) + [
        name for child in tree for name in calls(child)]


def validate(source, root):
    forms = parse(source)
    loads = [x[1].strip('"') for x in parse(root)]
    expected = ["tr3_c_snapshot_root.esk", "g3t_model_admission_extension.esk",
                "g3r_same_owner_hold_bridge_extension.esk",
                "g3r_held_o2_adopter_extension.esk",
                "g3r_same_owner_scope_extension.esk"]
    if [loads.index(x) for x in expected] != sorted(loads.index(x) for x in expected):
        raise ValueError("joint owner load order changed")
    bind = define(forms, "g3r-scope-bind!")
    for exact in ("(eq? model-entry (vector-ref record 9))",
                  "(eq? (vector-ref generator-entry 5) model-entry)",
                  "(eq? (vector-ref generator-entry 6) tokenizer)",
                  "(string=? (t2-wave2-tokenizer-fingerprint tokenizer) (vector-ref trainer-tokenizer 0))",
                  "(= (t2-wave2-tokenizer-vocab-size tokenizer) (vector-ref trainer-tokenizer 1))"):
        if not contains(bind, form(exact)):
            raise ValueError("original trainer/G3/tokenizer binding removed")
    enter = define(forms, "g3r-same-owner-scope-enter!")
    body = enter[2]
    if body[2] != form("(vector-set! g3r-scope-registry 0 next)"):
        raise ValueError("scope graph not rooted before lease publication")
    guard = body[3]
    if guard[0] != "guard" or not contains(guard[1], form("(g3r-scope-drain! scope caught)")):
        raise ValueError("scope cleanup handler missing")
    leased = guard[-1]
    if leased[0] != "let" or leased[1] != form("((leased (tr3-lease-snapshot-enter-internal! trainer parent)))"):
        raise ValueError("trainer lease acquisition changed")
    if leased[2] != form("(vector-set! (vector-ref scope 1) 0 leased)"):
        raise ValueError("trainer lease not rooted immediately")
    if leased[3] != form("(tr3-lease-snapshot-recheck-internal leased parent)"):
        raise ValueError("trainer idle recheck missing before native holds")
    if not contains(leased[5], form("(tr3-c-snapshot-controls! leased parent)")):
        raise ValueError("genuine trainer controls not captured")
    pair = leased[-1]
    if pair[0] != "let" or pair[1] != form("((pair (g3r-hold-pair-begin! generator (vector-ref scope 6) (vector-ref scope 7))))"):
        raise ValueError("native pair not derived from trainer owners")
    if pair[2] != form("(vector-set! (vector-ref scope 2) 0 pair)"):
        raise ValueError("native pair not rooted immediately")
    if not contains(pair, form("(g3r-hold-pair-check! pair operation)")):
        raise ValueError("post-acquisition pair check missing")
    if not contains(pair, form("(tr3-c-snapshot-controls! leased parent)")) or \
            not contains(pair, form("(vector-ref scope 10)")):
        raise ValueError("trainer epoch not compared after hold acquisition")
    if pair[-2] != form("(vector-set! scope 9 'live)"):
        raise ValueError("scope published before all rechecks")
    drain = define(forms, "g3r-scope-drain!")
    order = calls(drain)
    if order.index("g3r-hold-pair-end!") >= order.index("tr3-lease-snapshot-abort-internal!"):
        raise ValueError("trainer lease drained before native holds")
    if not contains(drain, form("(if (not first) (set! first caught))")):
        raise ValueError("first typed error not preserved")
    if not contains(drain, form("(g3r-holds-fail-stop)")):
        raise ValueError("uncertain trainer release no longer fails stop")
    if not contains(drain, form("(eq? (vector-ref pair 8) 'dead)")) or \
            not contains(drain, form("(eq? (vector-ref record 3) 'idle)")) or \
            not contains(drain, form("(eq? (vector-ref (vector-ref scope 8) 3) 'dead)")):
        raise ValueError("release lifecycle proof removed")
    if not contains(drain, form("(and (eq? (vector-ref scope 9) 'live) (not pair))")) or \
            not contains(drain, form("(and (eq? (vector-ref scope 9) 'live) (not record))")):
        raise ValueError("lost live authority can bypass fail stop")
    end = define(forms, "g3r-same-owner-scope-end!")
    accessor = define(forms, "g3r-same-owner-scope-pair!")
    if accessor[2] != form("(g3r-same-owner-scope-check! scope)"):
        raise ValueError("pair capability exposed without owner recheck")
    if not contains(end, form("(g3r-scope-drain! scope #f)")) or \
            "g3r-same-owner-scope-check!" in calls(end):
        raise ValueError("end can strand holds after a failed precheck")


class SameOwnerScopeSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text()
        cls.root = LOAD_ROOT.read_text()

    def test_contract(self):
        validate(self.source, self.root)

    def test_owner_and_order_mutations_fail(self):
        for old, new in (
            ("(eq? (vector-ref generator-entry 5) model-entry)", "#t"),
            ("(eq? (vector-ref generator-entry 6) tokenizer)", "#t"),
            ("(vector-set! g3r-scope-registry 0 next)", "#t"),
            ("(tr3-lease-snapshot-recheck-internal leased parent)", "#t"),
            ("(vector-set! (vector-ref scope 2) 0 pair)", "#t"),
            ("(g3r-hold-pair-end! pair)", "#t"),
            ("(eq? (vector-ref pair 8) 'dead)", "#t"),
        ):
            with self.subTest(old=old):
                self.assertEqual(self.source.count(old), 1)
                with self.assertRaises(ValueError):
                    validate(self.source.replace(old, new), self.root)

    def test_load_order_mutation_fails(self):
        mutated = self.root.replace(
            '(load "g3r_held_o2_adopter_extension.esk")\n'
            '(load "g3r_same_owner_scope_extension.esk")',
            '(load "g3r_same_owner_scope_extension.esk")\n'
            '(load "g3r_held_o2_adopter_extension.esk")')
        self.assertNotEqual(mutated, self.root)
        with self.assertRaises(ValueError):
            validate(self.source, mutated)


if __name__ == "__main__":
    unittest.main()
