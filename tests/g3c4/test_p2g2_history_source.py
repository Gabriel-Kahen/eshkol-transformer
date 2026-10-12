"""Source controls for the private P2/G1-G2 history prerequisite.

This does not substitute for compiled native and public runtime fault cuts.
"""

from importlib.machinery import SourceFileLoader
from pathlib import Path
from copy import deepcopy
import unittest

ROOT = Path(__file__).resolve().parents[2]
parse = SourceFileLoader(
    "history_source_forms", str(ROOT / "scripts/check-g3c4-p2g2-terminal-close.py")
).load_module()


def source(name: str) -> str:
    return (ROOT / name).read_text()


def final_predicate(text: str) -> list:
    return next(form for form in parse.forms(text)
                if isinstance(form, list) and form[0] == "define"
                and isinstance(form[1], list)
                and form[1][0] == "g3c4-public-p2g2-history-final?")[2]


def eq_walkers(text: str) -> dict:
    names = {"g3c4-public-g2-eq-member?", "g3c4-public-g2-eq-assoc"}
    return {form[1][0]: (form[1][1:], form[2])
            for form in parse.forms(text)
            if isinstance(form, list) and form[0] == "define"
            and isinstance(form[1], list) and form[1][0] in names}


def eval_fixed(node, env):
    """Execute the fixed predicate and its local eq? walkers from parsed AST."""
    if isinstance(node, str):
        if node in env:
            return env[node]
        if node == "#t":
            return True
        if node == "#f":
            return False
        if node.startswith("'"):
            return node[1:]
        return int(node)
    verb = node[0]
    if verb == "and":
        return all(eval_fixed(child, env) for child in node[1:])
    if verb == "not":
        return not eval_fixed(node[1], env)
    if verb == "cond":
        for clause in node[1:]:
            if clause[0] == "else" or eval_fixed(clause[0], env):
                return eval_fixed(clause[1], env)
        return False
    if verb == "let":
        name, bindings, body = node[1:]
        local = env.copy()
        local.update({key: eval_fixed(value, env) for key, value in bindings})
        local[name] = lambda *args: eval_fixed(
            body, {**local, **dict(zip((key for key, _ in bindings), args))})
        return eval_fixed(body, local)
    args = [eval_fixed(child, env) for child in node[1:]]
    if verb in env.get("__helpers__", {}):
        names, body = env["__helpers__"][verb]
        return eval_fixed(body, {**env, **dict(zip(names, args))})
    if verb in env and callable(env[verb]):
        return env[verb](*args)
    if verb == "null?":
        return not args[0]
    if verb == "car":
        return args[0][0]
    if verb == "cdr":
        return args[0][1:]
    if verb == "vector?":
        return isinstance(args[0], list)
    if verb == "vector-length":
        return len(args[0])
    if verb == "vector-ref":
        return args[0][args[1]]
    if verb == "eq?":
        return args[0] is args[1] or (
            isinstance(args[0], str) and args[0] == args[1])
    if verb == "=":
        return args[0] == args[1]
    raise AssertionError(f"unexpected predicate operation {verb}")


def final_fixture(first: int, second: int, count: int, eos: int):
    generator, input_shell, output = object(), object(), object()
    model, tokenizer, native, entry = object(), object(), object(), object()
    source_row, ticket = [None] * 12, [None] * 21
    source_row[0], source_row[1] = output, entry
    source_row[3], source_row[4], source_row[8] = generator, input_shell, "live"
    source_row[9:12] = [count, first, second]
    ticket[1], ticket[2], ticket[5], ticket[6] = generator, input_shell, output, entry
    ticket[10], ticket[12] = source_row, "live"
    ticket[13:16] = [count, first, second]
    generator_row, input_entry = [None] * 8, [None] * 12
    generator_row[3], generator_row[4], generator_row[5], generator_row[7] = (
        model, tokenizer, [None] * 6 + [eos], "live")
    input_entry[2], input_entry[3] = "live", native
    snapshot = [first, second, count, count + 2, 1, 2, 3, 4]
    env = dict(source=source_row, ticket=ticket, **{
        "generator-row": generator_row, "input-entry": input_entry,
        "model": model, "tokenizer": tokenizer, "generator": generator,
        "input": input_shell, "output": output, "eos": eos,
        "snapshot": snapshot, "g3c4-public-g2-output-rows": [[source_row]],
        "g3c4-public-g2-tickets": [[ticket]],
        "__helpers__": eq_walkers(source("native/g3c4_public_g2_result_extension.esk")),
    })
    return env


class HistorySource(unittest.TestCase):
    def setUp(self) -> None:
        self.c = source("src/eshkol_transformer/g3c4_model_owner.c")
        self.h = source("src/eshkol_transformer/g3c4_context_internal.h")
        self.input = source("native/g3c4_public_model_input_extension.esk")
        self.history = source("native/g3c4_public_g2_result_extension.esk")
        self.build = source("scripts/build-e1b-consumer.sh")
        self.package_test = source("scripts/test-g3g-manual-public.sh")

    def validate(self, c=None, h=None, input_source=None, history=None,
                 build=None) -> None:
        c = self.c if c is None else c
        h = self.h if h is None else h
        input_source = self.input if input_source is None else input_source
        history = self.history if history is None else history
        build = self.build if build is None else build
        copy = c.index("et_i64_tensor_copy_from_v1(\n              input->tensor")
        marker = c.index("input->exact_p2_constructor = 1u;")
        enroll = c.index("et_g3c4_enroll_input(input);", copy)
        assert copy < marker < enroll
        assert "if (require_p2)" in c[copy:marker]
        assert "sizeof(et_g3c4_input_internal) == 72u" in c
        assert "sizeof(et_g3c4_input_internal) == 48u" in c
        getter = c[c.index("int64_t et_g3c4_private_input_p2_original_snapshot_v1("):
                   c.index("int64_t et_g3c4_private_p2g2_terminal_snapshot_v1(")]
        for required in (
            "input->exact_p2_constructor != 1u", "destination_bytes != 24",
            "et_m3_private_i64_unborrowed_v1", "et_i64_tensor_borrow_begin_v1",
            "et_i64_tensor_borrow_view_v1", "view->shape[0] != 1u",
            "view->shape[1] != 2u", "words[0] != input->original_p2_ids[0]",
            "words[1] != input->original_p2_ids[1]",
            "et_i64_tensor_borrow_end_v1", "et_g3c4_terminal_snapshot_alias",
            "et_g3c4_error_restore_internal(first)",
        ):
            assert required in getter, required
        assert getter.count("input->exact_p2_constructor != 1u") == 2
        assert getter.count("et_g3c4_terminal_snapshot_alias(") == 2
        assert getter.index("et_i64_tensor_borrow_end_v1") < getter.index(
            "memcpy((unsigned char *)destination_header + 8u, encoded")
        assert "memset(input->original_p2_ids, 0" in c
        assert "input->exact_p2_constructor = 0u;" in c
        assert "et_g3c4_private_input_p2_original_snapshot_v1" in h
        assert "-DET_G3C4_P2_ORIGINAL_SNAPSHOT_PRIVATE" in build
        assert "et_g3c4_private_input_p2_original_snapshot_v1" in self.package_test
        assert "input_p2_original_snapshot" in self.package_test
        assert "(g3c4-public-p2g2-input-create-authority encoded)" in input_source
        assert "(lambda (encoded) (exit 134))" in input_source
        forms = parse.forms(history)
        assert forms and forms[-1][0] == "define"
        parsed_route = next(form for form in forms
                            if isinstance(form, list) and form[:2] ==
                            ["define", "g3c4-public-p2g2-history-route"])
        capture_let = next(form for form in parse.walk(parsed_route)
                           if isinstance(form, list) and form and
                           form[0] == "let*" and len(form) > 1 and
                           isinstance(form[1], list) and form[1] and
                           isinstance(form[1][0], list) and
                           form[1][0][0] == "shell")
        assert [binding[0] for binding in capture_let[1][-2:]] == [
            "descriptor", "authority"]
        assert len(capture_let) == 3
        assert capture_let[2][0] == "let"
        assert [binding[0] for binding in capture_let[2][1]] == [
            "next-history", "next-outputs"]
        dispatch = next(form for form in parse.walk(parsed_route)
                        if isinstance(form, list) and form and
                        form[0] == "cond" and any(
                            isinstance(clause, list) and clause and
                            isinstance(clause[0], list) and
                            clause[0][:2] == ["eq?", "action"]
                            for clause in form[1:]))
        assert [clause[0] for clause in dispatch[1:]] == [
            ["eq?", "action", "'input-issue"],
            ["eq?", "action", "'capture"],
            ["eq?", "action", "'admit"], "else"]
        assert [len(clause) for clause in dispatch[1:]] == [3, 2, 2, 2]
        route = history[history.index("(define g3c4-public-p2g2-history-route"):]
        for required in (
            "(let ((input-issuers '()) (history-issuers '())",
            "(secret (vector 'g3c4-public-p2g2-input-creator))",
            "(if (not (eq? third secret))",
            "'input-issue encoded created secret #f",
            "(generation-tensor-release-internal! created)",
            "(issued-outputs '())", "(>= input-count 64)",
            "(< history-count 64)", "(eq? phase 'live)",
            "(set! phase 'admitting)", "(eq? phase 'admitting)",
            "(set! phase 'spent)", "(set! phase 'live)",
            "(g3c4-public-g2-output-live output operation)",
            "g3c4-public-g2-tickets 5 output operation",
            "(g3c4-public-g2-source-snapshot output operation)",
            "(g3c4-public-p2g2-native-input-ids",
            "(t1-shell-read encoded 0)", "(t1-shell-read encoded 1)",
            "(not (g3c4-public-g2-eq-member? output issued-outputs))",
            "(raise caught)",
            "(g3c4-public-g2-eq-assoc first history-issuers)",
            "(= (vector-ref snapshot 0) eos)",
            "(not (= (vector-ref snapshot 0) eos))",
        ):
            assert required in route, required
        for index in range(8):
            assert f"(= (vector-ref current {index}) " in route
        issue = route[route.index("((eq? action 'input-issue)"):
                      route.index("((eq? action 'capture)")]
        issuer = issue[issue.index("(lambda (candidate candidate-entry"):
                       issue.index(";; The pair retains")]
        assert "t1-shell-" not in issuer
        assert "t1-wave1-tensor-admitted?" not in issuer
        assert "(cons input (cons encoded authority))" in issue
        assert issue.index("(t1-shell-read encoded 1)") < issue.index(
            "(ids (g3c4-public-p2g2-native-input-ids")
        capture = route[route.index("((eq? action 'capture)"):
                        route.index("((eq? action 'admit)")]
        final_read = "(final-snapshot\n                      (g3c4-p2g2-terminal-snapshot-private output))"
        assert capture.index("((cdr (cdr input-issuer)) input input-entry") < capture.index(final_read)
        assert capture.index(final_read) < capture.index("(set! history-issuers next-history)")
        admitted = capture[capture.index("(let ((current\n                                (g3c4-p2g2-terminal-snapshot-private output)))"):
                           capture.index("(set! phase 'spent)")]
        for forbidden in ("g3c4-public-g2-output-live", "g3c4-public-g2-source-snapshot",
                          "(cdr (cdr input-issuer))", "g3c4-public-p2g2-native-input-ids",
                          "t1-shell-"):
            assert forbidden not in admitted, forbidden
        final = history[history.index("(define (g3c4-public-p2g2-history-final?"):
                        history.index("(define g3c4-public-p2g2-history-route")]
        for forbidden in ("g3c4-public-g2-output-live", "g3c4-p2g2-terminal-snapshot-private",
                          "g3c4-public-p2g2-native-input-ids", "t1-shell-"):
            assert forbidden not in final, forbidden
        assert route.index("(set! created") < route.index(
            "'input-issue encoded created secret #f")
        assert route.index("(descriptor\n                      (vector p0 p1") < route.index(
            "(authority\n                      (lambda")
        assert route.index("(guard (caught\n                        (#t (if begun") < route.index(
            "((cdr issuer) 'begin second third fourth)")
        assert all(word not in route for word in (
            "g3c4-native-sample", "g3c4-native-generator-seed",
            "g3c4-native-generator-rng", "t1-tokenizer-decode"))

    def test_actual_source(self) -> None:
        self.validate()

    def test_local_eq_walkers_preserve_identity_and_association_pair(self) -> None:
        helpers = eq_walkers(self.history)
        self.assertEqual(set(helpers), {"g3c4-public-g2-eq-member?",
                                        "g3c4-public-g2-eq-assoc"})
        self.assertNotIn("memq", self.history)
        self.assertNotIn("assq", self.history)
        first, equal_copy = [1], [1]
        env = {"__helpers__": helpers, "value": first,
               "entries": [equal_copy, first]}
        member = ["g3c4-public-g2-eq-member?", "value", "entries"]
        self.assertTrue(eval_fixed(member, env))
        env["entries"] = [equal_copy]
        self.assertFalse(eval_fixed(member, env))
        env["entries"] = []
        self.assertFalse(eval_fixed(member, env))
        pair = [first, object()]
        env["entries"] = [[equal_copy, object()], pair]
        assoc = ["g3c4-public-g2-eq-assoc", "value", "entries"]
        self.assertIs(eval_fixed(assoc, env), pair)
        env["entries"] = [[equal_copy, object()]]
        self.assertFalse(eval_fixed(assoc, env))

    def test_fixed_predicate_rejects_equal_but_unrooted_rows(self) -> None:
        predicate = final_predicate(self.history)
        env = final_fixture(first=41, second=92, count=2, eos=7)
        self.assertTrue(eval_fixed(predicate, env))
        env["g3c4-public-g2-output-rows"] = [[deepcopy(env["source"])]]
        self.assertFalse(eval_fixed(predicate, env))
        env["g3c4-public-g2-output-rows"] = [[env["source"]]]
        env["g3c4-public-g2-tickets"] = [[deepcopy(env["ticket"])]]
        self.assertFalse(eval_fixed(predicate, env))

    def test_capture_publication_is_not_a_let_binding(self) -> None:
        malformed = self.history.replace(
            '"P2/G2 history verb is invalid")))))))',
            '"P2/G2 history verb is invalid"))))))', 1).replace(
                "                 ((cdr issuer) 'admit second third fourth)))))",
                "                 ((cdr issuer) 'admit second third fourth))))))", 1)
        with self.assertRaises(AssertionError):
            self.validate(history=malformed)

    def test_admit_is_a_peer_dispatch_clause(self) -> None:
        malformed = self.history.replace(
            '                 shell))))\n', '                 shell)))\n', 1).replace(
                "                 ((cdr issuer) 'admit second third fourth)))))",
                "                 ((cdr issuer) 'admit second third fourth))))))", 1)
        with self.assertRaises(AssertionError):
            self.validate(history=malformed)

    def test_constructor_kind_and_same_handle_drift_controls(self) -> None:
        for old in ("input->exact_p2_constructor = 1u;",
                    "words[0] != input->original_p2_ids[0]",
                    "et_m3_private_i64_unborrowed_v1"):
            with self.assertRaises((AssertionError, ValueError)):
                self.validate(c=self.c.replace(old, "/* omitted */"))
        with self.assertRaises(AssertionError):
            start = self.c.index("int64_t et_g3c4_private_input_p2_original_snapshot_v1(")
            end = self.c.index("int64_t et_g3c4_private_p2g2_terminal_snapshot_v1(")
            cut = self.c[start:end]
            cut = cut.replace("input->exact_p2_constructor != 1u ||", "", 1)
            self.validate(c=self.c[:start] + cut + self.c[end:])

    def test_spend_and_output_origin_controls(self) -> None:
        for old in ("(set! phase 'admitting)", "(set! phase 'spent)",
                    "(not (g3c4-public-g2-eq-member? output issued-outputs))",
                    "g3c4-public-g2-tickets 5 output operation",
                    "(= (vector-ref current 7) r3)",
                    "(g3c4-public-g2-eq-assoc first history-issuers)"):
            with self.assertRaises(AssertionError):
                prefix, route = self.history.split(
                    "(define g3c4-public-p2g2-history-route", 1)
                self.validate(history=prefix +
                    "(define g3c4-public-p2g2-history-route" +
                    route.replace(old, "#t"))

    def test_creation_cleanup_and_private_gate(self) -> None:
        with self.assertRaises(AssertionError):
            self.validate(input_source=self.input.replace(
                "(g3c4-public-p2g2-input-create-authority encoded)", "#t", 1))
        with self.assertRaises(AssertionError):
            self.validate(history=self.history.replace(
                "(if (not (eq? third secret))", "(if #f", 1))
        with self.assertRaises(AssertionError):
            self.validate(build=self.build.replace(
                "-DET_G3C4_P2_ORIGINAL_SNAPSHOT_PRIVATE", "", 1))

    def test_post_creation_t1_release_is_not_an_admission_gate(self) -> None:
        # Native original IDs, not T1 liveness, authenticate later history.
        marker = "(eq? (g3c4-input-entry-live input operation)\n                                 entry)"
        bad = self.history.replace(marker,
            "(= (t1-shell-read encoded 0) id0)\n                            " + marker, 1)
        with self.assertRaises(AssertionError):
            self.validate(history=bad)

    def test_no_late_input_provider_after_terminal_validation(self) -> None:
        # The D source ordering allowed a late provider to change output.
        spend = "(set! phase 'spent)"
        late = "((cdr (cdr input-issuer)) input input-entry (vector-ref input-entry 3))\n                              "
        with self.assertRaises(AssertionError):
            self.validate(history=self.history.replace(spend, late + spend, 1))
        final_read = "(final-snapshot\n                      (g3c4-p2g2-terminal-snapshot-private output))"
        with self.assertRaises((AssertionError, ValueError)):
            self.validate(history=self.history.replace(final_read,
                "(final-snapshot snapshot)", 1))

    def test_actual_fixed_predicate_eos_and_two_generation_slot_map(self) -> None:
        predicate = final_predicate(self.history)
        eos = final_fixture(first=7, second=0, count=1, eos=7)
        two = final_fixture(first=41, second=92, count=2, eos=7)
        self.assertNotEqual(eos["snapshot"][0], eos["snapshot"][3])
        self.assertNotEqual(two["snapshot"][0], two["snapshot"][3])
        self.assertTrue(eval_fixed(predicate, eos))
        self.assertTrue(eval_fixed(predicate, two))
        two["source"][11] = 93
        two["ticket"][15] = 93
        self.assertFalse(eval_fixed(predicate, two))
        two["source"][11] = 92
        two["ticket"][15] = 92
        self.assertTrue(eval_fixed(predicate, two))  # valid retry
        two["source"][10] = 44
        two["ticket"][14] = 44
        self.assertFalse(eval_fixed(predicate, two))

    def test_fixed_predicate_slot_swap_and_omission_mutants(self) -> None:
        original = final_predicate(self.history)
        good = final_fixture(first=41, second=92, count=2, eos=7)
        bad = deepcopy(original)
        for comparison in bad[1:]:
            if (comparison[0] == "=" and comparison[1] ==
                    ["vector-ref", "source", "10"]):
                comparison[2][2] = "3"
        self.assertFalse(eval_fixed(bad, good))
        changed = final_fixture(first=41, second=92, count=2, eos=7)
        changed["source"][11] = 93
        changed["ticket"][15] = 93
        dropped = [original[0]] + [item for item in original[1:]
            if item[1] not in (["vector-ref", "source", "11"],
                               ["vector-ref", "ticket", "15"])]
        self.assertTrue(eval_fixed(dropped, changed))
        self.assertFalse(eval_fixed(original, changed))


if __name__ == "__main__":
    unittest.main()
