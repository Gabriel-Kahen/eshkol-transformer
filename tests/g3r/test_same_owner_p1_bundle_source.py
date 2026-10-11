"""Owned source-state checks for the private P1 snapshot and one-use C2 bundle.

This does not execute compiled Eshkol or establish provider coexistence.
"""
from pathlib import Path
import unittest

from tests.c2.test_handler_order import parse
from tests.g3r.test_same_owner_o2_adoption_source import failed_cut

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "native/g3r_same_owner_o2_adoption_extension.esk"


def helpers(source):
    route = next(x for x in parse(source) if isinstance(x, list) and
                 x[:2] == ["define", "g3r-save-o2-adoption-route"])
    return dict(route[2][2][1])


def take_once(source, records, shell, authenticated=True, topology_ok=True,
              pending=False):
    """Interpret the parsed take lambda with owned state and failure cuts."""
    body = helpers(source)["take-for-c2"][2:]
    env = {"shell": shell, "issued": records, "pending": pending,
           "quarantine": False, "transfer-pending": False}
    events = []

    def evaluate(node, scope):
        if not isinstance(node, list):
            if node == "#t":
                return True
            if node == "#f":
                return False
            if isinstance(node, str) and node.isdigit():
                return int(node)
            return scope.get(node, node)
        op, *args = node
        if op == "quote":
            return args[0]
        if op == "if":
            return evaluate(args[1], scope) if evaluate(args[0], scope) else \
                (evaluate(args[2], scope) if len(args) > 2 else None)
        if op in ("or", "and"):
            return (any if op == "or" else all)(evaluate(x, scope) for x in args)
        if op == "not":
            return not evaluate(args[0], scope)
        if op == "eq?":
            left, right = evaluate(args[0], scope), evaluate(args[1], scope)
            return left == right if isinstance(left, str) and isinstance(right, str) \
                else left is right
        if op == "equal?":
            return evaluate(args[0], scope) == evaluate(args[1], scope)
        if op == "=":
            return evaluate(args[0], scope) == evaluate(args[1], scope)
        if op == "vector?":
            return isinstance(evaluate(args[0], scope), list)
        if op == "vector-length":
            return len(evaluate(args[0], scope))
        if op == "vector-ref":
            return evaluate(args[0], scope)[evaluate(args[1], scope)]
        if op == "vector-set!":
            evaluate(args[0], scope)[evaluate(args[1], scope)] = evaluate(args[2], scope)
            return None
        if op == "set!":
            env[args[0]] = evaluate(args[1], scope)
            return None
        if op == "car":
            return evaluate(args[0], scope)[0]
        if op == "cdr":
            return evaluate(args[0], scope)[1:]
        if op == "null?":
            return evaluate(args[0], scope) == []
        if op == "cond":
            for branch in args:
                if branch[0] == "else" or evaluate(branch[0], scope):
                    value = None
                    for statement in branch[1:]:
                        value = evaluate(statement, scope)
                    return value
            return None
        if op in ("let", "let*"):
            if isinstance(args[0], str):
                name, assignments, *statements = args
                def recur(rest):
                    frame = dict(scope)
                    frame[name] = recur
                    frame[assignments[0][0]] = rest
                    value = None
                    for statement in statements:
                        value = evaluate(statement, frame)
                    return value
                return recur(evaluate(assignments[0][1], scope))
            frame = dict(scope)
            for name, expression in args[0]:
                frame[name] = evaluate(expression, frame if op == "let*" else scope)
            value = None
            for statement in args[1:]:
                value = evaluate(statement, frame)
            return value
        if op == "guard":
            try:
                value = None
                for statement in args[1:]:
                    value = evaluate(statement, scope)
                return value
            except ValueError as error:
                env[args[0][0]] = error
                for statement in args[0][1][1:]:
                    evaluate(statement, env)
                raise
        if op == "require-owned":
            events.append(op)
            if not authenticated:
                raise ValueError("foreign shell")
            return True
        if op == "state-dict-c2-parameter-topology-internal":
            events.append(op)
            if not topology_ok:
                raise ValueError("invalid P1")
            return object()
        if op == "g3r-save-o2-fail":
            raise ValueError("admission denied")
        if op == "raise":
            raise evaluate(args[0], scope)
        raise AssertionError("unexpected parsed operation: " + op)

    try:
        value = None
        for statement in body:
            value = evaluate(statement, env)
        return value, env, events
    except ValueError:
        return None, env, events


class PrivateP1Bundle(unittest.TestCase):
    def setUp(self):
        self.source = SOURCE.read_text()

    def test_live_hold_capture_and_certain_cleanup(self):
        text = self.source[self.source.index("(adopt\n"):
                           self.source.index("(require-owned\n")]
        positions = [text.index(s) for s in (
            "(g3r-save-begin", "(vector-set! cell 1 hold)",
            "(let ((p1 (module-state-dict model)))", "(vector-set! cell 13 p1)",
            "(g3r-save-capture hold)", "(g3r-save-issue hold)",
            "(g3r-save-claim hold state)", "(g3r-save-end hold)")]
        self.assertEqual(positions, sorted(positions))
        failure = self.source[self.source.index("(drain-failed!\n"):
                              self.source.index("(adopt\n")]
        self.assertLess(failure.index("(= drain 2)"),
                        failure.index("(release-p1! cell)"))
        self.assertIn("(begin (release-p1! cell) #t)", failure)
        release = helpers(self.source)["release-p1!"]
        self.assertIn("state-dict-release!", str(release))
        self.assertIn("exit", str(release))
        mutated = text.replace("(let ((p1 (module-state-dict model)))",
                               "(let ((p1 #f))", 1)
        with self.assertRaises(ValueError):
            mutated.index("(let ((p1 (module-state-dict model)))")

    def test_one_use_authentication_and_no_spend_on_refusal(self):
        shell, p1, controls, words = object(), object(), object(), list(range(18))
        bundle = [p1, shell, controls, words, "prepared"]
        record = [shell, object(), object(), object(), words, controls,
                  object(), object(), object(), 7, p1, "issued", bundle,
                  words.copy()]
        output, state, events = take_once(self.source, [record], shell)
        self.assertIs(output, bundle)
        self.assertEqual((record[11], bundle[4]), ("spent", "transferred"))
        self.assertEqual(events, ["require-owned",
                                  "state-dict-c2-parameter-topology-internal"])
        output, _, events = take_once(self.source, [record], shell)
        self.assertIsNone(output)
        self.assertEqual(events, ["require-owned"])
        record[11], bundle[4] = "issued", "prepared"
        output, state, events = take_once(self.source, [record], shell,
                                          topology_ok=False)
        self.assertIsNone(output)
        self.assertEqual((record[11], bundle[4]), ("issued", "prepared"))
        self.assertFalse(state["transfer-pending"])
        output, state, _ = take_once(self.source, [record], shell,
                                     authenticated=False)
        self.assertIsNone(output)
        self.assertFalse(state["transfer-pending"])
        self.assertEqual(record[11], "issued")
        bundle[0] = object()
        output, _, _ = take_once(self.source, [record], shell)
        self.assertIsNone(output)
        self.assertEqual(record[11], "issued")
        bundle[0] = p1
        words[7] ^= 1
        output, _, _ = take_once(self.source, [record], shell)
        self.assertIsNone(output)
        self.assertEqual(record[11], "issued")
        words[7] ^= 1
        output, _, _ = take_once(self.source, [record], shell, pending=True)
        self.assertIsNone(output)
        self.assertEqual(record[11], "issued")

    def test_clone_failure_before_candidate_releases_hold_only(self):
        result, cell, quarantine, calls = failed_cut(
            self.source, "held", 2, p1_present=False)
        self.assertTrue(result)
        self.assertIsNone(quarantine)
        self.assertIsNone(cell[13])
        self.assertEqual(calls, ["g3r-save-end", "g3r-save-drain"])


if __name__ == "__main__":
    unittest.main()
