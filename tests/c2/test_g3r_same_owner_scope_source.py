"""Mutation-sensitive ownership order for the uncompiled private scope."""
from pathlib import Path
import unittest

from tests.c2.test_handler_order import form, parse


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "native/g3r_same_owner_scope_extension.esk"
LOAD_ROOT = ROOT / "native/g3r_same_owner_scope_root.esk"
AUTHORITY = ROOT / "native/g3r_same_owner_t2_authority_extension.esk"
T2_REGISTRY = ROOT / "internal/t2/lib/transformer/tokenizer_internal.esk"
CONSTRUCTOR = ROOT / "native/g3t_generator_constructor_extension.esk"
DECODER = ROOT / "native/g3c4_t1_output_decode_extension.esk"
T2_ROOT = ROOT / "native/t2_wave2_root.esk"


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


def validate(source, root, authority, registry, constructor, decoder):
    forms = parse(source)
    loads = [x[1].strip('"') for x in parse(root)]
    expected = ["tr3_c_snapshot_root.esk",
                "g3r_same_owner_t2_authority_extension.esk",
                "g3t_model_admission_extension.esk",
                "g3r_same_owner_hold_bridge_extension.esk",
                "g3r_held_o2_adopter_extension.esk",
                "g3r_same_owner_scope_extension.esk"]
    if [loads.index(x) for x in expected] != sorted(loads.index(x) for x in expected):
        raise ValueError("joint owner load order changed")
    if "tokenizer_internal.esk" in root or "t1_wave1_root.esk" in root:
        raise ValueError("second tokenizer authority loaded")
    if parse(authority) != [form("(define t1-private-entry t2-private-entry)"),
                            form("(define-syntax t1-private-raise (syntax-rules () ((_ category operation message) (t2-private-raise category operation message))))")]:
        raise ValueError("G3 alias must use the single T2 registry and typed raise")
    if not contains(define(parse(registry), "t2-private-entry"),
                    form("(eq? value (car (car remaining)))")):
        raise ValueError("T2 entry must authenticate shell identity")
    if not contains(define(parse(constructor), "g3t-generator-create"),
                    form("(t1-private-entry tokenizer)")) or \
            not contains(define(parse(decoder), "t1-private-g3-decode-raw-into!"),
                         form("(t1-private-entry tokenizer)")):
        raise ValueError("G3 leaf no longer checks tokenizer authority")
    for leaf in (constructor, decoder):
        if "(t1-core-tokenizer? core)" not in leaf or "(eq? (vector-ref core 1) 'raw)" not in leaf:
            raise ValueError("G3 leaf no longer checks the raw T1 core profile")
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
    unlink = define(forms, "g3r-scope-unlink!")
    if not contains(unlink, form("(eq? scope (car remaining))")) or \
            not contains(unlink, form("(set-cdr! previous (cdr remaining))")) or \
            not contains(unlink, form("(vector-set! g3r-scope-registry 0 (cdr remaining))")) or \
            not contains(unlink, form("(g3r-holds-fail-stop)")):
        raise ValueError("scope must unlink exact head/interior entry or fail stop")
    if drain[-3:] != [form("(vector-set! scope 9 'dead)"),
                      form("(g3r-scope-unlink! scope)"), "first"]:
        raise ValueError("completed scope retained or unlinked before release proof")
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
        cls.authority = AUTHORITY.read_text()
        cls.registry = T2_REGISTRY.read_text()
        cls.constructor = CONSTRUCTOR.read_text()
        cls.decoder = DECODER.read_text()

    def validate(self, source=None, root=None, authority=None, registry=None,
                 constructor=None, decoder=None):
        validate(source if source is not None else self.source,
                 root if root is not None else self.root,
                 authority if authority is not None else self.authority,
                 registry if registry is not None else self.registry,
                 constructor if constructor is not None else self.constructor,
                 decoder if decoder is not None else self.decoder)

    def test_contract(self):
        self.validate()

    def test_feature_on_source_root_closure(self):
        self.assertIn('(load "t2_wave2_root.esk")',
                      (ROOT / "native/d2_wave2_root.esk").read_text())
        self.assertIn('(load "d2_wave2_root.esk")',
                      (ROOT / "native/tr3_c_snapshot_root.esk").read_text())
        self.assertIn('(load "transformer/tokenizer_internal.esk")',
                      T2_ROOT.read_text())
        for loaded in (x[1].strip('"') for x in parse(self.root)):
            self.assertTrue((ROOT / "native" / loaded).is_file(), loaded)
        self.assertNotIn('(load "t1_wave1_root.esk")', self.root)

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
                    self.validate(source=self.source.replace(old, new))

    def test_load_order_mutation_fails(self):
        mutated = self.root.replace(
            '(load "g3r_held_o2_adopter_extension.esk")\n'
            '(load "g3r_same_owner_scope_extension.esk")',
            '(load "g3r_same_owner_scope_extension.esk")\n'
            '(load "g3r_held_o2_adopter_extension.esk")')
        self.assertNotEqual(mutated, self.root)
        with self.assertRaises(ValueError):
            self.validate(root=mutated)

    def test_foreign_and_bit_equal_distinct_tokenizer_mutations_fail(self):
        with self.assertRaisesRegex(ValueError, "shell identity"):
            self.validate(registry=self.registry.replace(
                "(eq? value (car (car remaining)))",
                "(equal? value (car (car remaining)))"))
        with self.assertRaisesRegex(ValueError, "single T2 registry"):
            self.validate(authority=self.authority.replace(
                "(define t1-private-entry t2-private-entry)",
                "(define (t1-private-entry value) (t1-private-entry value))"))
        with self.assertRaisesRegex(ValueError, "authority"):
            self.validate(constructor=self.constructor.replace(
                "(t1-private-entry tokenizer)", "(list tokenizer core)", 1))

    def test_completed_scope_retention_mutations_fail(self):
        for old, new in (
            ("\n  (g3r-scope-unlink! scope)\n", "\n  #t\n"),
            ("(eq? scope (car remaining))", "(equal? scope (car remaining))"),
            ("(set-cdr! previous (cdr remaining))", "#t"),
            ("(vector-set! g3r-scope-registry 0 (cdr remaining))", "#t"),
        ):
            with self.subTest(old=old):
                self.assertEqual(self.source.count(old), 1)
                with self.assertRaises(ValueError):
                    self.validate(source=self.source.replace(old, new))


if __name__ == "__main__":
    unittest.main()
