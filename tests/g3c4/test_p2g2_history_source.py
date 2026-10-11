"""Source controls for the private P2/G1-G2 history prerequisite.

This does not substitute for compiled native and public runtime fault cuts.
"""

from importlib.machinery import SourceFileLoader
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
parse = SourceFileLoader(
    "history_source_forms", str(ROOT / "scripts/check-g3c4-p2g2-terminal-close.py")
).load_module()


def source(name: str) -> str:
    return (ROOT / name).read_text()


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
            "(not (memq output issued-outputs))",
            "(raise caught)", "(assq first history-issuers)",
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
                    "(not (memq output issued-outputs))",
                    "g3c4-public-g2-tickets 5 output operation",
                    "(= (vector-ref current 7) r3)",
                    "(assq first history-issuers)"):
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


if __name__ == "__main__":
    unittest.main()
