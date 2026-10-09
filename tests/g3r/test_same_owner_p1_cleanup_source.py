"""Execute the parsed P1 release and adoption catch paths with owned faults."""
from pathlib import Path
import unittest

from tests.c2.test_handler_order import parse
from tests.g3r.test_same_owner_o2_adoption_source import failed_cut

SOURCE = Path(__file__).resolve().parents[2] / \
    "native/g3r_same_owner_o2_adoption_extension.esk"


def route(source):
    form = next(f for f in parse(source) if isinstance(f, list) and
                f[:2] == ["define", "g3r-save-o2-adoption-route"])
    return dict(form[2][2][1])


def release_cut(source, result):
    helper = route(source)["release-p1!"]
    owner = object()
    cell = [None] * 14
    cell[13] = owner
    env = {"cell": cell, "quarantine": False}
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
        if op == "let":
            frame = dict(scope)
            for name, value in args[0]:
                frame[name] = run(value, scope)
            value = None
            for statement in args[1:]:
                value = run(statement, frame)
            return value
        if op == "if":
            return run(args[1], scope) if run(args[0], scope) else \
                (run(args[2], scope) if len(args) > 2 else None)
        if op == "begin":
            value = None
            for statement in args:
                value = run(statement, scope)
            return value
        if op == "not":
            return not run(args[0], scope)
        if op == "vector-ref":
            return run(args[0], scope)[run(args[1], scope)]
        if op == "vector-set!":
            run(args[0], scope)[run(args[1], scope)] = run(args[2], scope)
            return None
        if op == "set!":
            env[args[0]] = run(args[1], scope)
            return None
        if op == "state-dict-release!":
            self_owner = run(args[0], scope)
            assert self_owner is owner
            calls.append("release")
            if isinstance(result, Exception):
                raise result
            return result
        if op == "guard":
            try:
                value = None
                for statement in args[1:]:
                    value = run(statement, scope)
                return value
            except Exception:
                for statement in args[0][1][1:]:
                    run(statement, scope)
                raise
        if op == "exit":
            raise SystemExit(run(args[0], scope))
        raise AssertionError("unexpected parsed P1 cleanup operation: " + op)

    try:
        run(helper[2], env)
        status = 0
    except SystemExit as exited:
        status = exited.code
    return status, cell, env["quarantine"], calls


class RaisedValue(Exception):
    def __init__(self, value):
        self.value = value


def catch_cut(source, first_error=False, original_value=None):
    adopt = route(source)["adopt"]
    guards = []
    def visit(node):
        if isinstance(node, list):
            if node[:1] == ["guard"] and "['drain-failed!', 'cell']" in str(node[1]):
                guards.append(node)
            for child in node:
                visit(child)
    visit(adopt)
    assert len(guards) == 1
    handler = guards[0][1]
    original = ValueError("P1 provider clone failed") if original_value is None \
        else original_value
    cell = [None] * 14
    cell[1] = object()
    first = [first_error, 4, 2, 7, 19]
    env = {"cell": cell, "first": first, "pending": cell,
           "caught": original, "native_latest": "original"}
    calls = []

    def run(node, scope):
        if not isinstance(node, list):
            if node == "#f":
                return False
            if node == "#t":
                return True
            return int(node) if node.isdigit() else scope.get(node, node)
        op, *args = node
        if op == "if":
            return run(args[1], scope) if run(args[0], scope) else \
                (run(args[2], scope) if len(args) > 2 else None)
        if op == "and":
            return all(run(x, scope) for x in args)
        if op == "not":
            return not run(args[0], scope)
        if op == "null?":
            return run(args[0], scope) is None
        if op == "vector-ref":
            return run(args[0], scope)[run(args[1], scope)]
        if op == "set!":
            env[args[0]] = run(args[1], scope)
            return None
        if op == "guard":
            for statement in args[1:]:
                run(statement, scope)
            return None
        if op == "drain-failed!":
            calls.append("drain")
            env["native_latest"] = "changed by native cleanup"
            return True
        if op == "raise":
            raised = run(args[0], scope)
            if isinstance(raised, BaseException):
                raise raised
            raise RaisedValue(raised)
        if op == "g3r-save-o2-raise-first":
            calls.append("first-native-error")
            raise RuntimeError("first native error")
        raise AssertionError("unexpected parsed catch operation: " + op)

    try:
        for statement in handler[1][1:]:
            run(statement, env)
    except Exception as error:
        return error, env, calls, original
    raise AssertionError("handler failed to rethrow")


class P1Cleanup(unittest.TestCase):
    def setUp(self):
        self.source = SOURCE.read_text()

    def test_release_success_false_and_exception(self):
        status, cell, quarantine, calls = release_cut(self.source, True)
        self.assertEqual((status, cell[13], quarantine, calls),
                         (0, False, False, ["release"]))
        for failed in (False, ValueError("provider release failed")):
            with self.subTest(failed=failed):
                status, cell, quarantine, calls = release_cut(self.source, failed)
                self.assertEqual(status, 134)
                self.assertIs(quarantine, cell)
                self.assertIsNotNone(cell[13])
                self.assertEqual(calls, ["release"])

    def test_clone_error_keeps_original_after_native_drain(self):
        error, env, calls, original = catch_cut(self.source)
        self.assertIs(error, original)
        self.assertIs(env["pending"], False)
        self.assertEqual(calls, ["drain"])
        self.assertEqual(env["native_latest"], "changed by native cleanup")
        error, env, calls, original = catch_cut(
            self.source, original_value=False)
        self.assertIsInstance(error, RaisedValue)
        self.assertIs(error.value, False)
        self.assertEqual(calls, ["drain"])
        error, env, calls, original = catch_cut(self.source, first_error=True)
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(calls, ["drain", "first-native-error"])

    def test_uncertain_drain_retains_p1_and_first_error(self):
        first = [True, 5, 23, 7, 912]
        result, cell, quarantine, calls = failed_cut(
            self.source, "pending", 1, first=first)
        self.assertFalse(result)
        self.assertIs(quarantine, cell)
        self.assertIsNotNone(cell[13])
        self.assertEqual(first, [True, 5, 23, 7, 912])
        self.assertEqual(calls, ["g3r-save-cancel", "g3r-save-end",
                                 "g3r-save-drain"])


if __name__ == "__main__":
    unittest.main()
