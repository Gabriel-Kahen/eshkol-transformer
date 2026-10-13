"""Execute the private fixed-word predicate and check the transfer boundary."""
from pathlib import Path
from copy import deepcopy
import unittest

from tests.c2.test_handler_order import parse
from tests.g3r.test_same_owner_save_bit_contract import roster


ROOT = Path(__file__).resolve().parents[2]
CALLER = ROOT / "native/g3r_same_owner_save_caller_extension.esk"
ADOPTION = ROOT / "native/g3r_same_owner_o2_adoption_extension.esk"


def definition(source, name):
    return next(form for form in parse(source)
                if isinstance(form, list) and form[:2] == ["define", name])


def validate_cleanup_root(caller):
    """Require a rooted adoption return before any post-adoption operation."""
    outer = caller[2]
    assert outer[0] == "let"
    assert ["shell-cell", ["vector", "#f"]] in outer[1]
    assert len(outer[2:]) == 1
    guard = outer[2]
    assert guard[0] == "guard"
    assert guard[1] == ["caught", ["#t",
        ["let", [["owned-shell", ["vector-ref", "shell-cell", "0"]]],
         ["if", "owned-shell",
          ["guard", ["cleanup", ["#t", ["exit", "134"]]],
           ["g3r-save-o2-abandon-internal!", "owned-shell"]]]],
        ["raise", "caught"]]]
    assert len(guard[2:]) == 2
    publish, continuation = guard[2:]
    assert publish == ["vector-set!", "shell-cell", "0",
        ["g3r-save-o2-adopt-held-internal", "generator", "model",
         "resolved", "tokenizer", "dataset", "optimizer", "capsule"]]
    assert continuation[0:2] == ["let", [["shell", ["vector-ref", "shell-cell", "0"]]]]
    assert continuation[2] == ["set!", "bundle",
                               ["g3r-save-o2-take-for-c2-internal", "shell"]]


def check_words(source, words, policy):
    body = definition(source, ["g3r-save-fixed-words!", "words", "generator"])[2:]

    def evaluate(node, scope):
        if not isinstance(node, list):
            if node == "#f":
                return False
            if node == "#t":
                return True
            if isinstance(node, str) and node.lstrip("-").isdigit():
                return int(node)
            return scope.get(node, node)
        op, *args = node
        if op == "quote":
            return args[0]
        if op == "begin":
            return sequence(args, scope)
        if op == "let*":
            local = dict(scope)
            for name, value in args[0]:
                local[name] = evaluate(value, local)
            return sequence(args[1:], local)
        if op == "let":
            if isinstance(args[0], str):
                name, bindings, *statements = args
                def recur(*values):
                    local = dict(scope)
                    local[name] = recur
                    for (binding, _), value in zip(bindings, values):
                        local[binding] = value
                    return sequence(statements, local)
                return recur(*(evaluate(value, scope) for _, value in bindings))
            local = dict(scope)
            for name, value in args[0]:
                local[name] = evaluate(value, scope)
            return sequence(args[1:], local)
        if op == "if":
            return evaluate(args[1], scope) if evaluate(args[0], scope) else \
                (evaluate(args[2], scope) if len(args) > 2 else None)
        if op == "and":
            return all(evaluate(arg, scope) for arg in args)
        if op == "or":
            return any(evaluate(arg, scope) for arg in args)
        if op == "not":
            return not evaluate(args[0], scope)
        if op == "vector?":
            return isinstance(evaluate(args[0], scope), list)
        if op == "vector-length":
            return len(evaluate(args[0], scope))
        if op == "vector-ref":
            return evaluate(args[0], scope)[evaluate(args[1], scope)]
        if op == "g3t-generator-find":
            return [None] * 7 + [policy]
        if op == "m3t-exact-i64?":
            value = evaluate(args[0], scope)
            return type(value) is int and -(1 << 63) <= value < (1 << 63)
        if op == "eq?":
            return evaluate(args[0], scope) == evaluate(args[1], scope)
        if op == "=":
            return evaluate(args[0], scope) == evaluate(args[1], scope)
        if op == "<":
            return evaluate(args[0], scope) < evaluate(args[1], scope)
        if op == "<=":
            return evaluate(args[0], scope) <= evaluate(args[1], scope)
        if op == ">=":
            return evaluate(args[0], scope) >= evaluate(args[1], scope)
        if op == "+":
            return sum(evaluate(arg, scope) for arg in args)
        if op == "g3r-save-fixed-fail":
            raise ValueError(evaluate(args[0], scope))
        if op == "check":
            return scope["check"](evaluate(args[0], scope))
        raise AssertionError("unexpected fixed-word operation: " + op)

    def sequence(nodes, scope):
        result = None
        for node in nodes:
            result = evaluate(node, scope)
        return result

    return sequence(body, {"words": words, "generator": object()})


def authenticate(source, record, shell, bundle, controls_before):
    route = definition(source, "g3r-save-o2-adoption-route")
    helper = dict(route[2][2][1])["authenticate-bundle"]
    issued = [record]

    def evaluate(node, scope):
        if not isinstance(node, list):
            if node == "#f":
                return False
            if node == "#t":
                return True
            if isinstance(node, str) and node.isdigit():
                return int(node)
            return scope.get(node, node)
        op, *args = node
        if op == "quote":
            return args[0]
        if op == "if":
            return evaluate(args[1], scope) if evaluate(args[0], scope) else \
                (evaluate(args[2], scope) if len(args) > 2 else None)
        if op == "or":
            return any(evaluate(arg, scope) for arg in args)
        if op == "and":
            return all(evaluate(arg, scope) for arg in args)
        if op == "not":
            return not evaluate(args[0], scope)
        if op == "eq?":
            left, right = evaluate(args[0], scope), evaluate(args[1], scope)
            return left == right if isinstance(left, str) and isinstance(right, str) \
                else left is right
        if op == "equal?":
            return evaluate(args[0], scope) == evaluate(args[1], scope)
        if op == "vector?":
            return isinstance(evaluate(args[0], scope), list)
        if op == "vector-length":
            return len(evaluate(args[0], scope))
        if op == "vector-ref":
            return evaluate(args[0], scope)[evaluate(args[1], scope)]
        if op == "=":
            return evaluate(args[0], scope) == evaluate(args[1], scope)
        if op == "null?":
            return not evaluate(args[0], scope)
        if op == "car":
            return evaluate(args[0], scope)[0]
        if op == "cdr":
            return evaluate(args[0], scope)[1:]
        if op == "cond":
            for branch in args:
                if branch[0] == "else" or evaluate(branch[0], scope):
                    return sequence(branch[1:], scope)
        if op in ("let", "let*"):
            if isinstance(args[0], str):
                name, bindings, *statements = args
                def recur(*values):
                    local = dict(scope)
                    local[name] = recur
                    for (binding, _), value in zip(bindings, values):
                        local[binding] = value
                    return sequence(statements, local)
                return recur(*(evaluate(value, scope) for _, value in bindings))
            local = dict(scope)
            for name, value in args[0]:
                local[name] = evaluate(value, local if op == "let*" else scope)
            return sequence(args[1:], local)
        if op == "require-owned":
            return True
        if op == "g3r-save-o2-check-roots":
            if record[5] != controls_before:
                raise ValueError("nested captured controls changed")
            return True
        if op == "state-dict-c2-parameter-topology-internal":
            return [[], [], []]
        if op == "g3r-save-o2-fail":
            raise ValueError("bundle denied")
        raise AssertionError("unexpected bundle authentication operation: " + op)

    def sequence(nodes, scope):
        result = None
        for node in nodes:
            result = evaluate(node, scope)
        return result

    return sequence(helper[2:], {"shell": shell, "bundle": bundle,
                                 "issued": issued, "pending": False,
                                 "quarantine": False,
                                 "transfer-pending": False})


def check_topology(source, topology, expected):
    body = definition(source, ["g3r-save-fixed-topology!", "p1"])[2:]

    def evaluate(node, scope):
        if not isinstance(node, list):
            if node == "#t":
                return True
            if isinstance(node, str) and node.isdigit():
                return int(node)
            return scope.get(node, node)
        op, *args = node
        if op == "quote":
            return args[0]
        if op == "let*":
            local = dict(scope)
            for name, value in args[0]:
                local[name] = evaluate(value, local)
            return sequence(args[1:], local)
        if op == "let":
            name, bindings, *statements = args
            def recur(*values):
                local = dict(scope)
                local[name] = recur
                for (binding, _), value in zip(bindings, values):
                    local[binding] = value
                return sequence(statements, local)
            return recur(*(evaluate(value, scope) for _, value in bindings))
        if op == "if":
            return evaluate(args[1], scope) if evaluate(args[0], scope) else \
                (evaluate(args[2], scope) if len(args) > 2 else None)
        if op == "and":
            return all(evaluate(arg, scope) for arg in args)
        if op == "not":
            return not evaluate(args[0], scope)
        if op == "null?":
            return not evaluate(args[0], scope)
        if op == "pair?":
            return isinstance(evaluate(args[0], scope), list) and \
                bool(evaluate(args[0], scope))
        if op == "equal?" or op == "=":
            return evaluate(args[0], scope) == evaluate(args[1], scope)
        if op == "vector-ref":
            return evaluate(args[0], scope)[evaluate(args[1], scope)]
        if op == "car":
            return evaluate(args[0], scope)[0]
        if op == "cdr":
            return evaluate(args[0], scope)[1:]
        if op == "+":
            return evaluate(args[0], scope) + evaluate(args[1], scope)
        if op == "state-dict-c2-parameter-topology-internal":
            return topology
        if op == "g3r-save-fixed-fail":
            raise ValueError("roster denied")
        if op == "check":
            return scope["check"](*(evaluate(arg, scope) for arg in args))
        raise AssertionError("unexpected topology operation: " + op)

    def sequence(nodes, scope):
        result = None
        for node in nodes:
            result = evaluate(node, scope)
        return result

    return sequence(body, {"p1": object(), "m3t-paths": expected[0],
                           "m3t-shapes": expected[1]})


class SameOwnerCaller(unittest.TestCase):
    def setUp(self):
        self.caller = CALLER.read_text()
        self.adoption = ADOPTION.read_text()

    def test_fixed_word_policy_and_malformed_negatives(self):
        policy = ["diagnostic-c2", 0, 1065353216, 256, 1065353216, 1, False]
        words = [0, 1065353216, 256, 1065353216, 1, -1,
                 1, 2, 3, 4, 8, 0, 1, 1, 0, 0, 0]
        self.assertTrue(check_words(self.caller, words, policy))
        for index, value in ((0, 2), (0, "bad"), (3, 255),
                             (6, "bad"), (10, 1 << 63)):
            changed = words.copy()
            changed[index] = value
            with self.subTest(index=index), self.assertRaises(ValueError):
                check_words(self.caller, changed, policy)
        # The divergent ledger had an 18-word version-prefixed image.
        with self.assertRaises(ValueError):
            check_words(self.caller, [1] + words, policy)
        with self.assertRaises(ValueError):
            check_words(self.caller, words[:-1], policy)
        for index, value in ((11, 1), (12, 0), (12, 2),
                             (13, 0), (14, 1), (15, 1), (16, 1)):
            changed = words.copy()
            changed[index] = value
            with self.subTest(history=index), self.assertRaises(ValueError):
                check_words(self.caller, changed, policy)
        no_generation_budget = words.copy()
        no_generation_budget[4] = 0
        no_generation_budget[11] = 9
        no_generation_budget[14] = 1
        no_generation_budget[12] = 2
        changed_policy = policy.copy()
        changed_policy[5] = 0
        with self.assertRaises(ValueError):
            check_words(self.caller, no_generation_budget, changed_policy)

    def test_exact_fourteen_primary_roster_and_tied_alias(self):
        contract = (ROOT / "docs/g3/G3_R_SAME_OWNER_SAVE_BIT_WITNESS_CONTRACT.md").read_text()
        rows = roster(contract)
        self.assertEqual(len(rows), 14)
        paths = [[f'"{part}"' for part in path.split("/")] for _, path, _ in rows]
        shapes = [[str(dim) for dim in shape] for _, _, shape in rows]
        aliases = [[["\"head\"", "\"weight\""],
                    ["\"token_embedding\"", "\"weight\""]]]
        topology = [paths, shapes, aliases]
        self.assertTrue(check_topology(self.caller, topology, (paths, shapes)))
        for section, index, changed in ((0, 10, ['"wrong"']),
                                        (1, 4, ["8", "4"])):
            bad = deepcopy(topology)
            bad[section][index] = changed
            with self.subTest(section=section, index=index), self.assertRaises(ValueError):
                check_topology(self.caller, bad, (paths, shapes))
        bad = deepcopy(topology)
        bad[2] = []
        with self.assertRaises(ValueError):
            check_topology(self.caller, bad, (paths, shapes))

    def test_caller_orders_reauthentication_before_c2_claim(self):
        text = self.caller.split("(define (g3r-save-c2-capture-internal", 1)[1]
        markers = ["(g3r-save-o2-adopt-held-internal",
                   "(g3r-save-o2-take-for-c2-internal shell)",
                   "(g3r-save-o2-authenticate-bundle-internal shell bundle)",
                   "(g3r-save-fixed-topology!", "(g3r-save-fixed-controls!",
                   "(g3r-save-fixed-words!", "(vector-set! result 1",
                   "(vector c2-training-state-compose-tag",
                   "(c2-training-state-compose-internal request)",
                   "(vector-set! result 0"]
        positions = [text.index(marker) for marker in markers]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("(vector-ref (vector-ref bundle 2) 0)", text)
        self.assertIn("(g3r-save-o2-abandon-internal! owned-shell)", text)
        self.assertIn("(equal? (vector-ref result 1) (vector-ref bundle 3))", text)

    def test_private_record_holds_independent_words_and_reverse_cleanup(self):
        self.assertIn("(vector-set! record 13 (vector-copy words))", self.adoption)
        self.assertIn("(equal? (vector-ref bundle 3)\n                                              (vector-ref record 13))",
                      self.adoption)
        cleanup = self.adoption.split("(abandon\n", 1)[1].split("(retry\n", 1)[0]
        self.assertLess(cleanup.index("o2-public-optimizer-state-release!"),
                        cleanup.index("state-dict-release!"))
        self.assertIn("(exit 134)", cleanup)

    def test_post_take_word_control_and_bundle_mutation_denied(self):
        shell, p1 = object(), object()
        controls = [[1, [2, 3], [b"current", b"epoch"], [b"config", "id"],
                     [7, 2, 0]], 3]
        before = deepcopy(controls)
        words = list(range(17))
        bundle = [p1, shell, controls, words, "transferred"]
        row = [shell, object(), None, None, None, None, "live", False]
        record = [shell, row, row[1], object(), words, controls,
                  object(), object(), object(), 2, p1, "spent", bundle,
                  words.copy()]
        self.assertTrue(authenticate(self.adoption, record, shell, bundle, before))
        words[7] ^= 1
        with self.assertRaises(ValueError):
            authenticate(self.adoption, record, shell, bundle, before)
        words[7] ^= 1
        controls[0][2][0] = b"drift"
        with self.assertRaises(ValueError):
            authenticate(self.adoption, record, shell, bundle, before)
        controls[0][2][0] = b"current"
        with self.assertRaises(ValueError):
            authenticate(self.adoption, record, shell, bundle.copy(), before)
        bundle[4] = "retired"
        with self.assertRaises(ValueError):
            authenticate(self.adoption, record, shell, bundle, before)

    def test_c2_refusal_rethrows_first_error_after_one_abandon(self):
        caller = definition(self.caller,
                            ["g3r-save-c2-capture-internal", "generator",
                             "model", "resolved", "tokenizer", "dataset",
                             "optimizer", "capsule"])
        validate_cleanup_root(caller)
        # The parsed handler has one lexical cleanup call, preserves the
        # original caught value, and fails stop if cleanup itself fails.
        guard = caller[2][2]
        self.assertEqual(str(guard).count("'g3r-save-o2-abandon-internal!'"), 1)
        self.assertLess(str(guard).index("'g3r-save-o2-abandon-internal!'"),
                        str(guard).index("'raise', 'caught'"))

    def test_cleanup_root_rejects_stale_or_late_publication(self):
        caller = definition(self.caller,
                            ["g3r-save-c2-capture-internal", "generator",
                             "model", "resolved", "tokenizer", "dataset",
                             "optimizer", "capsule"])
        stale = deepcopy(caller)
        stale[2][2][1][1][1][1][0][1] = "shell"
        with self.assertRaises(AssertionError):
            validate_cleanup_root(stale)

        late = deepcopy(caller)
        late[2][2][2:4] = late[2][2][3:1:-1]
        with self.assertRaises(AssertionError):
            validate_cleanup_root(late)

        unrooted = deepcopy(caller)
        unrooted[2][1][0][1] = "#f"
        with self.assertRaises(AssertionError):
            validate_cleanup_root(unrooted)


if __name__ == "__main__":
    unittest.main()
