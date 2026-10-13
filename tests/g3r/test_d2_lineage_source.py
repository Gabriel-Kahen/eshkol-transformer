"""Private D2/T2 lexical lineage source-state controls; compiled behavior is open."""
from pathlib import Path
import unittest

from tests.c2.test_handler_order import parse

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "native/g3r_d2_lineage_extension.esk"
PRIVATE_ROOT = ROOT / "native/g3r_d2_lineage_root.esk"
MANIFEST = ROOT / "native/g3r_d2_lineage_source_closure.txt"


def descendants(expr, tag):
    if isinstance(expr, list):
        if expr and expr[0] == tag:
            yield expr
        for child in expr:
            yield from descendants(child, tag)


def route(source):
    forms = parse(source)
    matches = [f for f in forms if isinstance(f, list) and
               f[:2] == ["define", "g3r-d2-lineage-route"]]
    if len(matches) != 1:
        raise ValueError("missing fixed lexical route")
    return matches[0]


def required_predicate(source):
    binding = route(source)[2][1][2][1]
    matches = [a for a in descendants(binding, "and") if
               ["eq?", "original", "tokenizer"] in a]
    if len(matches) != 1:
        raise ValueError("missing checked original identity")
    return matches[0]


def search_predicate(source):
    binding = route(source)[2][1][2][1]
    key = ["vector-ref", ["cdr", ["car", "records"]], "0"]
    matches = [a for a in descendants(binding, "eq?") if key in a and
               "dataset" in a]
    if len(matches) != 1:
        raise ValueError("search no longer uses issuer-held dataset")
    return matches[0]


def eval_pure(expr, env):
    """Evaluate only the reviewed source predicate with owned identity objects."""
    if isinstance(expr, str):
        if expr in env:
            return env[expr]
        if expr.isdecimal():
            return int(expr)
        raise ValueError(f"unexpected source atom: {expr}")
    op, *args = expr
    if op == "and":
        return all(eval_pure(a, env) for a in args)
    if op == "quote":
        return args[0]
    if op == "vector-ref":
        return eval_pure(args[0], env)[eval_pure(args[1], env)]
    if op == "car":
        return eval_pure(args[0], env)[0]
    if op == "cdr":
        return eval_pure(args[0], env)[1]
    if op == "d2-core-state-identity=?":
        return (eval_pure(args[0], env) is env["core"] and
                eval_pure(args[1], env) is env["decoded"] and
                env["cursor_valid"])
    if op in {"=", "<=", "string=?"}:
        a, b = (eval_pure(x, env) for x in args)
        return {"=": a == b, "<=": a <= b, "string=?": a == b}[op]
    if op == "eq?":
        a, b = (eval_pure(x, env) for x in args)
        return a is b or isinstance(a, str) and a == b
    raise ValueError(f"unexpected source predicate: {op}")


def completed_state():
    dataset, tokenizer, tokenizer_entry, state = (object() for _ in range(4))
    fingerprint, vocabulary = "original-fingerprint", 256
    core, decoded = [None] * 16, [None] * 16
    core[2], core[3], core[14], core[15] = fingerprint, vocabulary, 100, 7
    decoded[15] = 7
    row = [dataset, "live", tokenizer, tokenizer_entry,
           state, fingerprint, vocabulary, core]
    snapshot = [dataset, tokenizer, tokenizer_entry,
                state, fingerprint, vocabulary, core]
    return dict(dataset=dataset, tokenizer=tokenizer, original=tokenizer,
                row=row, snapshot=snapshot, state=state,
                tokenizer_entry=tokenizer_entry, **{"tokenizer-entry": tokenizer_entry}, core=core,
                fingerprint=fingerprint, vocabulary=vocabulary,
                decoded=decoded, cursor_valid=True)


def check_lineage(source: str, root: str, manifest: str) -> None:
    forms = parse(source)
    names = [f[1][0] for f in forms if isinstance(f, list) and len(f) > 2 and
             f[0] == "define" and isinstance(f[1], list)]
    if names != ["g3r-d2-lineage-check-capacity",
                 "g3r-d2-lineage-byte-core", "g3r-d2-lineage-open",
                 "g3r-d2-lineage-require", "g3r-d2-lineage-retire!"]:
        raise ValueError("private lineage surface differs")
    if any(isinstance(f, list) and f[:2] ==
           ["define", "g3r-d2-lineage-registry"] for f in forms) or \
            "g3r-d2-lineage-registry" in source:
        raise ValueError("completed-row authority escaped lexical scope")
    if parse(root) != [["load", '"cli3_generate_root.esk"'],
                       ["load", '"g3r_d2_lineage_extension.esk"']]:
        raise ValueError("same-aggregate load order differs")
    expected = ["native/g3r_d2_lineage_root.esk",
                *(ROOT / "native/cli3_generate_source_closure.txt").read_text().splitlines(),
                "native/g3r_d2_lineage_extension.esk"]
    if manifest.splitlines() != expected or len(set(expected)) != len(expected):
        raise ValueError("source closure differs")
    bridge = (ROOT / "native/cli3_generate_t2_bridge.esk").read_text()
    old_root = (ROOT / "native/cli3_generate_root.esk").read_text()
    if "(define t1-private-entry t2-private-entry)" not in bridge or \
            old_root.index('(load "cli3_generate_t2_bridge.esk")') > old_root.index(
                '(load "g3t_generator_constructor_extension.esk")'):
        raise ValueError("T2/T1 registry bridge changed")
    r = route(source)
    bindings = r[2][1]
    if r[2][0] != "let*" or [b[0] for b in bindings] != \
            ["registry", "open", "required", "retire"] or \
            bindings[0] != ["registry", ["vector", ["quote", []], "#f"]] or \
            r[2][2][0] != "lambda":
        raise ValueError("registry or routes are caller-replaceable")
    for text in ("(g3r-d2-lineage-route 'open config tokenizer)",
                 "(g3r-d2-lineage-route 'require dataset tokenizer)",
                 "(g3r-d2-lineage-route 'retire dataset tokenizer)",
                 "((eq? mode 'open) (open first second))",
                 "((eq? mode 'require) (required first second))",
                 "((eq? mode 'retire) (retire first second))"):
        if text not in source:
            raise ValueError("fixed wrapper/route changed")
    stages = ["(g3r-d2-lineage-check-capacity (vector-ref registry 0) operation)",
              "(tokenizer-entry (g3r-d2-lineage-byte-core tokenizer operation))",
              "(record (cons row snapshot))",
              "(next (cons record (vector-ref registry 0)))",
              "(vector-set! registry 1 pending)",
              "(d2-token-dataset-open config tokenizer)",
              "(vector-set! pending 2 shell)",
              "(d2-open-dataset-state shell operation)",
              "(vector-set! snapshot 0 shell)",
              "(vector-set! snapshot 6 core)",
              "(vector-set! row 1 'live)",
              "(vector-set! registry 0 next)",
              "(vector-set! registry 1 #f)"]
    positions = [source.rfind(x) if x == stages[-1] else source.find(x)
                 for x in stages]
    if -1 in positions or positions != sorted(positions):
        raise ValueError("same-call staged issuance/publication differs")
    for text in ("(guard (cleanup-failure (#t (exit 134)))",
                 "(d2-token-dataset-close! (vector-ref pending 2))",
                 "(raise caught)",
                 "(define g3r-d2-lineage-maximum-rows 64)"):
        if text not in source:
            raise ValueError("failure cleanup or quota differs")
    quota = source[source.index("(define (g3r-d2-lineage-check-capacity"):
                   source.index("(define (g3r-d2-lineage-byte-core")]
    if quota.index("((= capacity 0)") > quota.index("((null? remaining)"):
        raise ValueError("row limit boundary widened")
    auth = required_predicate(source)
    search = search_predicate(source)
    required = [["eq?", "original", "tokenizer"],
                ["eq?", ["vector-ref", "snapshot", "3"], "state"],
                ["eq?", ["vector-ref", "snapshot", "6"], "core"],
                ["eq?", ["vector-ref", "row", "0"], "dataset"],
                ["eq?", ["vector-ref", "row", "2"], "original"],
                ["eq?", ["vector-ref", "row", "4"], "state"],
                ["eq?", ["vector-ref", "row", "7"], "core"],
                ["d2-core-state-identity=?", "core", "decoded"]]
    genuine = completed_state()
    genuine["records"] = ((genuine["row"], genuine["snapshot"]), None)
    if (any(item not in auth for item in required) or
            not eval_pure(auth, genuine) or not eval_pure(search, genuine)):
        raise ValueError("issuer-held original snapshot not authenticated")
    if ("((eq? (vector-ref (cdr (car records)) 0) dataset)" not in source or
            "(let search ((records (vector-ref registry 0)))" not in source or
            "\n                      #t))\n                   (else (search (cdr records))))))))" not in source):
        raise ValueError("forged ordinary row or returned authority permitted")
    retire = source[source.index("(retire\n") : source.index("    (lambda (mode first second)")]
    order = ["(next (append (reverse prefix) (cdr records)))",
             "(required dataset tokenizer)",
             "(d2-token-dataset-close! dataset)",
             "(vector-set! registry 0 next)",
             "(vector-set! row 0 #f)",
             "(vector-set! snapshot 0 #f)"]
    points = [retire.find(x) for x in order]
    if -1 in points or points != sorted(points) or \
            "((eq? (vector-ref (cdr (car records)) 0) dataset)" not in retire or \
            "(#t (if (not (vector-ref state 2)) (exit 134))" not in retire or \
            "(if (vector-ref state 2) (exit 134))" not in retire:
        raise ValueError("retirement can prune failed/foreign row")
    d2 = (ROOT / "internal/d2/lib/d2_dataset.esk").read_text()
    native = (ROOT / "native/d2_native.c").read_text()
    opened = d2[d2.index("(define (d2-token-dataset-open config tokenizer)"):
                d2.index("(define (d2-token-dataset-next-batch dataset)")]
    if "(d2-make-dataset-shell state)" not in opened or \
            "(d2-native-dataset-open shell shuffle-count)" not in opened or \
            "et_d2_dataset_open_v1" not in native:
        raise ValueError("reviewed native D2 open path differs")


class D2LineageSource(unittest.TestCase):
    def setUp(self):
        self.source = EXT.read_text()
        self.root = PRIVATE_ROOT.read_text()
        self.manifest = MANIFEST.read_text()

    def check(self, **changes):
        d = dict(source=self.source, root=self.root, manifest=self.manifest)
        d.update(changes)
        check_lineage(**d)

    def test_exact_route(self):
        self.check()

    def test_genuine_completed_and_tampered_states(self):
        auth = required_predicate(self.source)
        state = completed_state()
        self.assertTrue(eval_pure(auth, state))
        search = search_predicate(self.source)
        state["records"] = ((state["row"], state["snapshot"]), None)
        self.assertTrue(eval_pure(search, state))
        ordinary = object()
        state["dataset"] = ordinary
        self.assertFalse(eval_pure(search, state))
        for mutation in (
            lambda e: e["row"].__setitem__(0, ordinary),
            lambda e: e["row"].__setitem__(2, object()),
            lambda e: e["row"].__setitem__(4, object()),
            lambda e: e["row"].__setitem__(7, object()),
            lambda e: e["row"].__setitem__(1, "retired"),
            lambda e: e.update(tokenizer=object()),
            lambda e: e["snapshot"].__setitem__(3, object()),
            lambda e: e["snapshot"].__setitem__(4, "alternate"),
            lambda e: e.update(cursor_valid=False),
        ):
            with self.subTest(mutation=mutation):
                e = completed_state()
                mutation(e)
                self.assertFalse(eval_pure(auth, e))

    def test_forged_or_copied_row_not_authority(self):
        self.check()
        self.assertNotIn("(define g3r-d2-lineage-registry", self.source)
        self.assertNotIn("(vector-ref row 7) row", self.source)
        self.assertIn("(record (cons row snapshot))", self.source)
        self.assertIn("((eq? (vector-ref (cdr (car records)) 0) dataset)", self.source)
        self.assertIn("\n                      #t))\n", self.source)
        with self.assertRaises(ValueError):
            self.check(source=self.source.replace(
                "(let search ((records (vector-ref registry 0)))",
                "(let search ((records '()))"))
        with self.assertRaises(ValueError):
            self.check(source=self.source.replace(
                "(eq? (vector-ref snapshot 3) state)", "#t"))

    def test_fixed_routes_and_aggregate(self):
        for before, after in (
            ("(registry (vector '() #f))", "(registry #f)"),
            ("((eq? mode 'require) (required first second))",
             "((eq? mode 'require) (open first second))"),
            ("(g3r-d2-lineage-route 'retire dataset tokenizer)",
             "(g3r-d2-lineage-route 'open dataset tokenizer)"),
        ):
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.check(source=self.source.replace(before, after))
        with self.assertRaises(ValueError):
            self.check(root=self.root.replace("cli3_generate_root.esk",
                                              "g3g_manual_package_root.esk"))
        with self.assertRaises(ValueError):
            self.check(manifest=self.manifest.replace(
                "native/cli3_generate_t2_bridge.esk\n", ""))

    def test_publication_and_abort_cuts(self):
        for before, after in (
            ("(vector-set! registry 0 next)", "#t"),
            ("(record (cons row snapshot))", "(record (cons row row))"),
            ("(d2-token-dataset-close! (vector-ref pending 2))", "#t"),
            ("(exit 134)", "#f"),
        ):
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.check(source=self.source.replace(before, after))

    def test_retire_requires_genuine_close_and_snapshot(self):
        for before, after in (
            ("(required dataset tokenizer)", "#t"),
            ("(next (append (reverse prefix) (cdr records)))",
             "(next (cdr records))"),
            ("(d2-token-dataset-close! dataset)", "#t"),
            ("(vector-set! snapshot 0 #f)", "#t"),
        ):
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.check(source=self.source.replace(before, after))

    def test_capacity_boundary_and_cursor(self):
        with self.assertRaises(ValueError):
            self.check(source=self.source.replace(
                "((= capacity 0)", "((null? remaining) #t)\n      ((= capacity 0)"))
        with self.assertRaises(ValueError):
            self.check(source=self.source.replace(
                "(d2-core-state-identity=? core decoded)", "#t"))
        with self.assertRaises(ValueError):
            self.check(source=self.source.replace(
                "(eq? original tokenizer)", "(equal? original tokenizer)"))


if __name__ == "__main__":
    unittest.main()
