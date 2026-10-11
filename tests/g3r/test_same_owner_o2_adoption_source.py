"""Private adoption source controls; compiled Eshkol behavior is still open."""
from pathlib import Path
import unittest

from tests.c2.test_handler_order import parse

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "native/g3r_same_owner_o2_adoption_extension.esk"
AGGREGATE = ROOT / "native/g3r_same_owner_o2_adoption_root.esk"
CLOSURE = ROOT / "native/g3r_same_owner_o2_adoption_source_closure.txt"


def section(text, first, last):
    start = text.index(first)
    return text[start:text.index(last, start)]


def ordered(text, markers):
    positions = [text.find(marker) for marker in markers]
    if -1 in positions or positions != sorted(positions):
        raise ValueError("private adoption order changed: " + repr(markers))


def validate(source, aggregate, closure):
    forms = parse(source)
    if parse(aggregate) != [
        ["load", '"g3r_training_controls_root.esk"'],
        ["load", '"g3r_same_owner_o2_adoption_extension.esk"'],
        ["load", '"g3r_same_owner_save_caller_extension.esk"'],
    ]:
        raise ValueError("private aggregate changed")
    expected = ["native/g3r_same_owner_o2_adoption_root.esk",
                *(ROOT / "native/g3r_training_controls_source_closure.txt")
                .read_text().splitlines(),
                "native/g3r_same_owner_o2_adoption_extension.esk",
                "native/g3r_same_owner_save_caller_extension.esk"]
    if closure.splitlines() != expected or len(set(expected)) != len(expected):
        raise ValueError("recursive source closure changed")
    actual = [(f[1], f[2], f[-1]) for f in forms
              if isinstance(f, list) and f[:1] == ["extern"]]
    names = ["begin", "capture", "snapshot_word", "o2_issue", "o2_claim",
             "o2_cancel", "check", "end", "cleanup_retry", "error_word",
             "drain_state"]
    expected_abi = [
        ("ptr" if name in ("begin", "o2_issue") else "i64",
         {"snapshot_word": "g3r-save-word", "cleanup_retry": "g3r-save-retry",
          "error_word": "g3r-save-error-word", "drain_state": "g3r-save-drain",
          "o2_issue": "g3r-save-issue", "o2_claim": "g3r-save-claim",
          "o2_cancel": "g3r-save-cancel"}.get(name, "g3r-save-" + name),
         "et_g3r_private_same_owner_save_" + name + "_v1") for name in names]
    if actual != expected_abi:
        raise ValueError("native scalar ABI changed")
    route = next((f for f in forms if isinstance(f, list) and
                  f[:2] == ["define", "g3r-save-o2-adoption-route"]), None)
    if route is None or route[2][0] != "let" or route[2][1] != [
        ["issued", ["quote", []]], ["pending", "#f"], ["quarantine", "#f"],
        ["transfer-pending", "#f"]]:
        raise ValueError("issuance authority is no longer lexical")
    quota = section(source, "(define (g3r-save-o2-capacity", ";;; roots:")
    ordered(quota, ["((= left 0)", "((null? rest)"])
    first = section(source, "(define (g3r-save-o2-copy-first!", "(define (g3r-save-o2-native-category")
    ordered(first, [f"(g3r-save-error-word hold {i})" for i in range(4)])
    if "(not (vector-ref first 0))" not in first:
        raise ValueError("cleanup may replace the first native error")
    roots = section(source, "(define (g3r-save-o2-check-roots", ";;; cell:")
    for evidence in ("g3r-generator-lineage-require", "g3r-controls-captured-copy",
                     "tr3-controls-route 'originals", "g3t-generator-find",
                     "g3t-model-entry-live", "o2-require-optimizer",
                     "d2-open-dataset-state", "tr3-lease-vector-eq?",
                     "tr3-lease-d2-idle!"):
        if evidence not in roots:
            raise ValueError("same-owner root recheck lost " + evidence)
    adopt = section(source, "(adopt\n", "(require-owned\n")
    ordered(adopt, ["(set! pending cell)", "(guard\n", "(g3r-save-begin",
                    "(vector-set! cell 1 hold)", "(if (null? hold)",
                    "(o2-copy-config-data", "(o2-make-identity",
                    "(next-o2 (cons row old-head))", "(record (vector",
                    "(next-issued (cons record issued))", "(g3r-save-capture hold)",
                    "(g3r-save-word hold index)", "(g3r-save-issue hold)",
                    "(vector-set! cell 2 state)", "(if (null? state)",
                    "(vector-set! cell 9 'pending)", "(g3r-save-claim hold state)",
                    "(vector-set! cell 9 'claimed)", "(let* ((status (g3r-save-end hold))",
                    "(drain (g3r-save-drain hold))", "(vector-set! row 6 'live)",
                    "(set! o2-states next-o2)", "(set! issued next-issued)"])
    claim = section(adopt, "(vector-set! cell 9 'pending)", "(g3r-save-claim hold state)")
    for evidence in ("(eq? old-head o2-states)", "(equal? (vector-ref record 6) config)",
                     "(equal? (vector-ref record 7) paths)",
                     "(equal? (vector-ref record 8) aliases)",
                     "(equal? config (vector-ref row 2))",
                     "(o2-copy-config-data operation", "(o2-copy-paths",
                     "(o2-copy-aliases", "(eq? (vector-ref row 1) #f)",
                     "(g3r-save-check hold)"):
        if evidence not in claim:
            raise ValueError("preclaim authentication lost " + evidence)
    ordered(claim, ["(g3r-save-o2-fail 'invalid-state 2",
                    "(g3r-save-check hold)",
                    "(g3r-save-o2-copy-first! first hold)",
                    "(g3r-save-o2-raise-first first)"])
    guard = section(adopt, "(guard\n", "(g3r-save-begin")
    if "(and (vector-ref cell 1)" not in guard or \
            "(not (null? (vector-ref cell 1)))" not in guard:
        raise ValueError("boxed null begin can enter native cleanup")
    fixed = adopt[adopt.index(";; All remaining writes are fixed") :]
    for forbidden in ("cons", "make-vector", "o2-require-optimizer", "raise",
                      "g3r-save-check", "g3r-save-drain"):
        if forbidden in fixed:
            raise ValueError("fallible work returned after native zero")
    failure = section(source, "(drain-failed!\n", "(adopt\n")
    for evidence in ("g3r-save-cancel hold", "g3r-save-end hold",
                     "(= drain 1)", "(= drain 2)", "(= drain 3)",
                     "(set! quarantine cell)", "(release-claimed! cell)", "(exit 134)"):
        if evidence not in failure:
            raise ValueError("failed adoption cleanup lost " + evidence)
    release = section(source, "(release-claimed!\n", "(drain-failed!\n")
    for status in ("(= status 0)", "(= status -2)", "(= status -1)", "(exit 134)"):
        if status not in release:
            raise ValueError("claimed state cleanup changed")
    owned = section(source, "(require-owned\n", "(retry\n")
    for evidence in ("(eq? row", "(o2-find-identity shell o2-states)",
                     "(eq? (vector-ref row 1) state)",
                     "(equal? (vector-ref row 2)",
                     "(equal? (vector-ref row 3)",
                     "(equal? (vector-ref row 4)",
                     "(= (vector-ref row 5)",
                     "(vector-ref record 9)",
                     "(eq? (vector-ref row 6) 'live)",
                     "(g3r-save-o2-check-roots roots)"):
        if evidence not in owned:
            raise ValueError("lexical issued-row validation lost " + evidence)


def failed_cut(source, phase, drain_state, release_ok=True, first=None,
               p1_present=True):
    """Execute the parsed failure helper with owned native/registry stand-ins."""
    form = next(f for f in parse(source) if isinstance(f, list) and
                f[:2] == ["define", "g3r-save-o2-adoption-route"])
    helper = next(x[1] for x in form[2][2][1] if x[0] == "drain-failed!")
    hold = object()
    cell = [None, hold, object(), None, None, None, None, None, None, phase,
            None, None, None, object() if p1_present else None]
    env = {"cell": cell, "hold": hold, "quarantine": None,
           "first": first}
    calls = []

    def run(node, scope):
        if not isinstance(node, list):
            if node == "#f":
                return False
            if node == "#t":
                return True
            if node.isdigit():
                return int(node)
            return scope.get(node, node)
        op, *args = node
        if op == "quote":
            return args[0]
        if op in ("let", "let*"):
            local = dict(scope)
            for name, value in args[0]:
                local[name] = run(value, local if op == "let*" else scope)
            result = None
            for body in args[1:]:
                result = run(body, local)
            return result
        if op == "if":
            return run(args[1], scope) if run(args[0], scope) else \
                (run(args[2], scope) if len(args) > 2 else None)
        if op == "cond":
            for branch in args:
                if branch[0] == "else" or run(branch[0], scope):
                    result = None
                    for body in branch[1:]:
                        result = run(body, scope)
                    return result
            return None
        if op == "begin":
            result = None
            for body in args:
                result = run(body, scope)
            return result
        if op == "vector-ref":
            return run(args[0], scope)[run(args[1], scope)]
        if op == "vector-set!":
            run(args[0], scope)[run(args[1], scope)] = run(args[2], scope)
            return None
        if op == "set!":
            env[args[0]] = run(args[1], scope)
            return None
        if op == "not":
            return not run(args[0], scope)
        if op == "null?":
            return run(args[0], scope) is None
        if op == "or":
            return any(run(item, scope) for item in args)
        if op in ("eq?", "="):
            return run(args[0], scope) == run(args[1], scope)
        if op in ("g3r-save-cancel", "g3r-save-end"):
            calls.append(op)
            return 0
        if op == "g3r-save-drain":
            calls.append(op)
            return drain_state
        if op == "release-p1!":
            if scope["cell"][13] is not None:
                calls.append(op)
            scope["cell"][13] = None
            return None
        if op == "release-claimed!":
            calls.append(op)
            return release_ok
        if op == "exit":
            raise SystemExit(run(args[0], scope))
        raise ValueError("unexpected failure-cut form: " + op)

    result = run(helper[2], env)
    return result, cell, env["quarantine"], calls


def native_preclaim_cut(source):
    """Execute the actual parsed final preclaim failure branch with fake words."""
    form = next(f for f in parse(source) if isinstance(f, list) and
                f[:2] == ["define", "g3r-save-o2-adoption-route"])
    adopt = next(x[1] for x in form[2][2][1] if x[0] == "adopt")
    matches = []
    def visit(node):
        if isinstance(node, list):
            if node[:1] == ["if"] and "['g3r-save-check', 'hold']" in str(node[1]):
                matches.append(node)
            for child in node:
                visit(child)
    visit(adopt)
    if len(matches) != 3:
        raise ValueError("expected P1, capture and final preclaim native checks")
    branch = matches[-1]
    words = (5, 23, 7, 912)
    recorded = []
    class NativeError(Exception):
        pass
    def run(node):
        if not isinstance(node, list):
            return int(node) if node.lstrip("-").isdigit() else node
        op, *args = node
        if op == "g3r-save-check":
            return -1
        if op == "=":
            return run(args[0]) == run(args[1])
        if op == "not":
            return not run(args[0])
        if op == "begin":
            for item in args:
                run(item)
            return None
        if op == "g3r-save-o2-copy-first!":
            recorded.append(words)
            return None
        if op == "g3r-save-o2-raise-first":
            raise NativeError(recorded[0] if recorded else None)
        raise ValueError("unexpected preclaim form " + op)
    try:
        if run(branch[1]):
            run(branch[2])
    except NativeError as error:
        return error.args[0], recorded
    raise ValueError("native preclaim failure was not raised")


class AdoptionSource(unittest.TestCase):
    def setUp(self):
        self.source = SOURCE.read_text()
        self.aggregate = AGGREGATE.read_text()
        self.closure = CLOSURE.read_text()

    def test_source_and_recursive_closure(self):
        validate(self.source, self.aggregate, self.closure)

    def test_preclaim_and_zero_tail_mutants(self):
        mutants = [
            ("(g3r-save-claim hold state)", "(g3r-save-claim hold #f)"),
            ("(vector-set! cell 2 state)", "(vector-set! cell 2 #f)"),
            ("(equal? (vector-ref record 6) config)", "#t"),
            ("(equal? (vector-ref record 7) paths)", "#t"),
            ("(equal? (vector-ref record 8) aliases)", "#t"),
            ("(eq? old-head o2-states)", "#t"),
            ("(not (null? (vector-ref cell 1)))", "#t"),
            ("(= (vector-ref row 5)", "(= (vector-ref record 9)"),
            ("(g3r-save-o2-copy-first! first hold)\n                                (g3r-save-o2-raise-first first)))\n                     (if (not (= (g3r-save-claim",
             "(g3r-save-o2-raise-first first))\n                     (if (not (= (g3r-save-claim"),
            ("(let* ((status (g3r-save-end hold))",
             "(let ((status (g3r-save-end hold))"),
            ("(vector-set! row 6 'live)", "(o2-require-optimizer 'late optimizer) (vector-set! row 6 'live)"),
        ]
        for before, after in mutants:
            with self.subTest(before=before), self.assertRaises(ValueError):
                if before == "(eq? old-head o2-states)":
                    head, tail = self.source.rsplit(before, 1)
                    changed = head + after + tail
                else:
                    changed = self.source.replace(before, after, 1)
                validate(changed, self.aggregate, self.closure)

    def test_wrong_aggregate_and_missing_member(self):
        with self.assertRaises(ValueError):
            validate(self.source, self.aggregate.replace("g3r_training_controls_root",
                                                         "g3r_g3_creation_lineage_root"), self.closure)
        with self.assertRaises(ValueError):
            validate(self.source, self.aggregate,
                     self.closure.replace("native/o2_wave2_extension.esk\n", ""))

    def test_owned_failure_cuts(self):
        result, cell, quarantine, calls = failed_cut(self.source, "pending", 2)
        self.assertTrue(result)
        self.assertIsNone(quarantine)
        self.assertFalse(cell[1])
        self.assertEqual(calls, ["g3r-save-cancel", "g3r-save-end", "g3r-save-drain", "release-p1!"])
        result, cell, quarantine, calls = failed_cut(self.source, "held", 1)
        self.assertFalse(result)
        self.assertIs(quarantine, cell)
        self.assertIsNotNone(cell[1])
        self.assertEqual(cell[9], "hold-cleanup")
        self.assertEqual(calls, ["g3r-save-end", "g3r-save-drain"])
        result, cell, quarantine, calls = failed_cut(self.source, "claimed", 3, False)
        self.assertFalse(result)
        self.assertIs(quarantine, cell)
        self.assertEqual(calls, ["g3r-save-end", "g3r-save-drain", "release-claimed!"])
        result, cell, quarantine, calls = failed_cut(self.source, "ended-claimed", 3)
        self.assertTrue(result)
        self.assertIsNone(quarantine)
        self.assertEqual(calls, ["g3r-save-drain", "release-claimed!", "release-p1!"])
        with self.assertRaises(SystemExit):
            failed_cut(self.source, "claimed", 2)

    def test_boxed_null_begin_skips_cleanup(self):
        form = next(f for f in parse(self.source) if isinstance(f, list) and
                    f[:2] == ["define", "g3r-save-o2-adoption-route"])
        adopt = next(x[1] for x in form[2][2][1] if x[0] == "adopt")
        conditions = []
        handlers = []
        def visit(node):
            if isinstance(node, list):
                if node[:1] == ["if"] and "['drain-failed!', 'cell']" in str(node[2]):
                    conditions.append(node[1])
                if node[:1] == ["guard"] and "['drain-failed!', 'cell']" in str(node[1]):
                    handlers.append(node[1][1][1:])
                for child in node:
                    visit(child)
        visit(adopt)
        self.assertEqual(len(conditions), 1)
        self.assertEqual(len(handlers), 1)
        class BoxedNull:
            pass
        null = BoxedNull()
        live = object()
        def evaluate(node, hold):
            if isinstance(node, list):
                op, *args = node
                if op == "and":
                    return all(evaluate(arg, hold) for arg in args)
                if op == "not":
                    return not evaluate(args[0], hold)
                if op == "null?":
                    return evaluate(args[0], hold) is null
                if op == "vector-ref":
                    self.assertEqual(args, ["cell", "1"])
                    return hold
                raise ValueError(op)
            if node == "#t":
                return True
            return node
        self.assertFalse(evaluate(conditions[0], null))
        self.assertTrue(evaluate(conditions[0], live))
        def caught_cut(hold):
            cell = [None, hold]
            first = [True, 5, 23, 7, 912]
            pending = [True]
            cleanup = []
            for expression in handlers[0]:
                if expression[0] == "if" and "drain-failed!" in str(expression):
                    if evaluate(expression[1], cell[1]):
                        cleanup.append("native-end")
                elif expression[:2] == ["set!", "pending"]:
                    pending[0] = False
                elif expression[0] == "if" and expression[1] == \
                        ["vector-ref", "first", "0"]:
                    self.assertTrue(first[0])
                    self.assertEqual(expression[2],
                                     ["g3r-save-o2-raise-first", "first"])
                    return cleanup, pending[0], tuple(first[1:])
                else:
                    raise ValueError("unexpected null-begin handler form")
            raise ValueError("first native error was not rethrown")
        self.assertEqual(caught_cut(null), ([], False, (5, 23, 7, 912)))
        self.assertEqual(caught_cut(live), (["native-end"], False,
                                            (5, 23, 7, 912)))

    def test_final_native_check_copies_typed_error(self):
        words, calls = native_preclaim_cut(self.source)
        self.assertEqual(words, (5, 23, 7, 912))
        self.assertEqual(calls, [words])
        first = [True, *words]
        result, cell, quarantine, cleanup = failed_cut(
            self.source, "pending", 2, first=first)
        self.assertTrue(result)
        self.assertIsNone(quarantine)
        self.assertEqual(cleanup,
                         ["g3r-save-cancel", "g3r-save-end", "g3r-save-drain", "release-p1!"])
        self.assertEqual(first, [True, *words])


if __name__ == "__main__":
    unittest.main()
