"""Execute the actual C2 compose form with controlled claim/rollback providers.

This bounded source check does not stand in for the compiled C2 owner gate.
"""

from copy import deepcopy
from pathlib import Path
import unittest

from tests.c2.test_handler_order import definition, lists, parse


SOURCE = Path(__file__).resolve().parents[2] / "native/c2_training_state_extension.esk"


class Raised(Exception):
    def __init__(self, value):
        self.value = value


class Stopped(BaseException):
    def __init__(self, status):
        self.status = status


def evaluate(node, env):
    if not isinstance(node, list):
        if node == "#t":
            return True
        if node == "#f":
            return False
        if node.lstrip("-").isdigit():
            return int(node)
        if node.startswith('"'):
            return node[1:-1]
        return env[node]
    head = node[0]
    if head == "quote":
        return node[1]
    if head == "let*":
        local = env.copy()
        for name, value in node[1]:
            local[name] = evaluate(value, local)
        return sequence(node[2:], local)
    if head == "guard":
        try:
            return sequence(node[2:], env)
        except Raised as caught:
            local = env.copy()
            local[node[1][0]] = caught.value
            for clause in node[1][1:]:
                if evaluate(clause[0], local) is not False:
                    return sequence(clause[1:], local)
            raise
    if head == "if":
        branch = node[2] if evaluate(node[1], env) is not False else (
            node[3] if len(node) > 3 else "#f")
        return evaluate(branch, env)
    if head == "begin":
        return sequence(node[1:], env)
    if head == "set!":
        env[node[1]] = evaluate(node[2], env)
        return True
    if head == "raise":
        raise Raised(evaluate(node[1], env))
    if head == "exit":
        raise Stopped(evaluate(node[1], env))
    if head == "not":
        return evaluate(node[1], env) is False
    if head == "or":
        for item in node[1:]:
            result = evaluate(item, env)
            if result is not False:
                return result
        return False
    if head == "and":
        result = True
        for item in node[1:]:
            result = evaluate(item, env)
            if result is False:
                return False
        return result
    function = env[head]
    return function(*(evaluate(arg, env) for arg in node[1:]))


def sequence(nodes, env):
    result = False
    for node in nodes:
        result = evaluate(node, env)
    return result


def compose_form():
    return definition({"source": parse(SOURCE.read_text())},
                      "c2-training-state-compose-internal")


def run(form, p1_claim, o2_claim, rollback):
    tag, p1, o2 = object(), object(), object()
    request = [tag, p1, o2, object(), "checkpoint-load", False]
    env = {
        "c2-training-state-compose-tag": tag,
        "c2-training-state-owner-token-tag": object(),
        "c2-training-state-states": [],
        "c2-training-state-controls-copy": lambda *_: [0, 1, 2, 3, [0, 0]],
        "state-dict-c2-parameter-topology-internal": lambda *_: object(),
        "c2-training-state-o2-preflight": lambda *_: object(),
        "c2-training-state-claim-p1!": p1_claim,
        "c2-training-state-claim-o2!": o2_claim,
        "c2-training-state-rollback-p1!": rollback,
        "c2-training-state-fail": lambda *_: (_ for _ in ()).throw(AssertionError()),
        "error": lambda *_: (_ for _ in ()).throw(Raised("shell")),
        "vector": lambda *values: list(values),
        "vector?": lambda value: isinstance(value, list),
        "vector-length": len,
        "vector-ref": lambda value, index: value[index],
        "vector-set!": lambda value, index, item: value.__setitem__(index, item),
        "symbol?": lambda value: isinstance(value, str),
        "=": lambda left, right: left == right,
        "eq?": lambda left, right: left is right or (
            isinstance(left, str) and isinstance(right, str) and left == right),
        "cons": lambda first, rest: [first, *rest],
    }
    return request, lambda: sequence(form[2:], {**env, "request": request})


class ComposeRollbackSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.form = compose_form()

    def test_success_consumes_and_publishes(self):
        calls = []
        request, invoke = run(
            self.form,
            lambda *_: calls.append("p1") or True,
            lambda *_: calls.append("o2") or True,
            lambda *_: calls.append("rollback") or True)
        self.assertIs(invoke(), request)
        self.assertEqual(calls, ["p1", "o2"])
        self.assertEqual(request[1:3], [False, False])
        self.assertEqual(request[5], "shell")

    def test_certain_rollback_preserves_original_false_and_roots(self):
        calls, owner = [], {"token": None}

        def claim(_, token):
            calls.append("p1")
            owner["token"] = token
            return True

        def fail_o2(*_):
            calls.append("o2")
            raise Raised(False)

        def rollback(_, token):
            calls.append("rollback")
            self.assertIs(owner["token"], token)
            owner["token"] = None
            return True

        request, invoke = run(self.form, claim, fail_o2, rollback)
        original = request[1:3]
        with self.assertRaises(Raised) as caught:
            invoke()
        self.assertIs(caught.exception.value, False)
        self.assertEqual(calls, ["p1", "o2", "rollback"])
        self.assertIsNone(owner["token"])
        self.assertEqual(request[1:3], original)
        self.assertIs(request[5], False)

    def test_certain_rollback_preserves_structured_identity(self):
        original_failure = object()

        def fail_o2(*_):
            raise Raised(original_failure)

        request, invoke = run(self.form, lambda *_: True, fail_o2,
                              lambda *_: True)
        with self.assertRaises(Raised) as caught:
            invoke()
        self.assertIs(caught.exception.value, original_failure)
        self.assertIsNot(request[1], False)
        self.assertIsNot(request[2], False)
        self.assertIs(request[5], False)

    def test_preclaim_failure_does_not_rollback(self):
        calls = []

        def fail_p1(*_):
            calls.append("p1")
            raise Raised(False)

        request, invoke = run(self.form, fail_p1,
                              lambda *_: calls.append("o2") or True,
                              lambda *_: calls.append("rollback") or True)
        with self.assertRaises(Raised) as caught:
            invoke()
        self.assertIs(caught.exception.value, False)
        self.assertEqual(calls, ["p1"])
        self.assertIsNot(request[1], False)
        self.assertIsNot(request[2], False)
        self.assertIs(request[5], False)

    def test_nontrue_claim_result_fail_stops_before_publication(self):
        for failed in ("p1", "o2"):
            with self.subTest(failed=failed):
                calls = []

                def claim_p1(*_):
                    calls.append("p1")
                    return failed != "p1"

                def claim_o2(*_):
                    calls.append("o2")
                    return failed != "o2"

                request, invoke = run(
                    self.form, claim_p1, claim_o2,
                    lambda *_: calls.append("rollback") or True)
                with self.assertRaises(Stopped) as stopped:
                    invoke()
                self.assertEqual(stopped.exception.status, 134)
                self.assertEqual(calls, ["p1"] if failed == "p1" else
                                 ["p1", "o2"])
                self.assertIsNot(request[1], False)
                self.assertIsNot(request[2], False)
                self.assertIs(request[5], False)

    def test_uncertain_rollback_fail_stops_once(self):
        for result in ("throw", "false"):
            with self.subTest(result=result):
                calls, owner = [], {"token": None}

                def claim(_, token):
                    calls.append("p1")
                    owner["token"] = token
                    return True

                def fail_o2(*_):
                    calls.append("o2")
                    raise Raised(False)

                def rollback(_, token):
                    calls.append("rollback")
                    self.assertIs(owner["token"], token)
                    if result == "throw":
                        raise Raised("rollback-defect")
                    return False

                request, invoke = run(self.form, claim, fail_o2, rollback)
                original = request[1:3]
                with self.assertRaises(Stopped) as stopped:
                    invoke()
                self.assertEqual(stopped.exception.status, 134)
                self.assertEqual(calls, ["p1", "o2", "rollback"])
                self.assertIsNotNone(owner["token"])
                self.assertEqual(request[1:3], original)
                self.assertIs(request[5], False)

    def test_fail_stop_guard_is_required(self):
        changed = deepcopy(self.form)
        outer = next(node for node in lists(changed)
                     if node[:1] == ["guard"] and node[1][0] == "rollback-defect")
        outer[1][1][1] = ["raise", "rollback-defect"]

        def fail(*_):
            raise Raised(False)

        def bad_rollback(*_):
            raise Raised("rollback-defect")

        request, invoke = run(changed, lambda *_: True, fail, bad_rollback)
        with self.assertRaises(Raised) as caught:
            invoke()
        self.assertEqual(caught.exception.value, "rollback-defect")
        self.assertIs(request[5], False)


if __name__ == "__main__":
    unittest.main()
