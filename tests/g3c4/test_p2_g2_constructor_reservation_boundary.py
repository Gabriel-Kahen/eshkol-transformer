"""Source preflight for preserving the accepted private reservation paths."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


def source(path):
    return (ROOT / path).read_text(encoding="utf-8")


def section(text, begin, end):
    start = text.index(begin)
    return text[start:text.index(end, start + len(begin))]


class ConstructorReservationBoundaryTests(unittest.TestCase):
    def assert_order(self, body, *steps):
        offset = 0
        for step in steps:
            found = body.find(step, offset)
            self.assertNotEqual(found, -1, f"missing or out of order: {step}")
            offset = found + len(step)

    def test_constructor_owns_all_eight_allocations(self):
        carriers = source("native/g3c4_p2g2_terminal_carriers_extension.esk")
        create = section(
            carriers,
            "(set! g3c4-p2g2-terminal-carriers-create",
            "(set! g3c4-p2g2-terminal-carriers-check",
        )
        self.assertRegex(create, r"\(lambda \(\)\s+\(let\*")
        allocations = (
            ("first-raw", 1), ("first-staging", 8),
            ("pending-raw", 2), ("pending-staging", 16),
            ("second-raw", 1), ("second-staging", 8),
            ("text-one", 1), ("text-two", 2),
        )
        self.assertEqual(
            re.findall(r"\(([\w-]+) \(make-bytevector (\d+) 0\)\)", create),
            [(name, str(length)) for name, length in allocations],
        )
        for slot, (name, _) in enumerate(allocations):
            with self.subTest(slot=slot):
                self.assertIn(f"(eq? (vector-ref bundle {slot}) {name})", create)
        self.assertIn("(extents (vector 9 16 10 24 9 16 9 10))", create)
        self.assertNotIn("(extern ", carriers)

    def test_existing_enrollment_owns_its_original_auxiliary(self):
        reserve = section(
            source("native/g3c4_output_envelope_extension.esk"),
            "(define (g3c4-output-reserve", "(define (g3c4-output-prepare",
        )
        self.assert_order(
            reserve, "(g3c4-output-reserve call-entry prompt-length)",
            "(if (vector-ref ledger 4)", "(raw (make-bytevector generated 0))",
            "(staging (make-bytevector (* generated 8) 0))",
            "(auxiliary (vector raw staging))",
            "(entry (vector shell 'output 'pending #f generator-entry",
            "model-entry tokenizer #f auxiliary #f #f",
            "(next (cons entry (vector-ref g3c4-registry 0)))",
            "(guard", "(g3c4-native-output-release native)",
            "(eq? (vector-ref ledger 4) entry)", "(vector-set! ledger 4 #f)",
            "(g3c4-entry-dead! entry)", "(raise caught)",
            "(vector-set! g3c4-registry 0 next)",
            "(vector-set! ledger 4 canonical)", "(g3c4-native-output-reserve",
            "(vector-set! canonical 3 created)",
        )
        self.assertNotRegex(reserve, r"\(vector-set!\s+(?:entry|canonical)\s+8\b")

    def test_abort_only_scope_has_separate_first_and_second_allocations(self):
        scope = source("native/g3c4_p2g2_protected_scope_extension.esk")
        self.assert_order(
            scope, "(define-syntax g3c4-with-p2g2-protected-call",
            "(prefix-raw (make-bytevector 1 0))",
            "(prefix-staging (make-bytevector 8 0))",
            "(g3c4-native-call-acquire native 2 2)",
        )
        pending = source("native/g3c4_p2g2_terminal_pending_anchor_extension.esk")
        self.assert_order(
            pending, "(set! g3c4-p2g2-terminal-pending-prepare",
            "(g3c4-p2g2-protected-reserve call)",
            "(second-raw (make-bytevector 1 0))",
            "(second-staging (make-bytevector 8 0))",
        )
        for legacy in (scope, pending):
            self.assertNotIn("g3c4-p2g2-terminal-carriers-create", legacy)
            self.assertNotIn("et_g3c4_private_p2g2_terminal_commit_v1", legacy)
        feature_off = source("tests/g3c4/p2_g2_protected_scope_feature_off_test.esk")
        self.assertNotIn("g3c4_p2g2_ownership_extension.esk", feature_off)
        self.assertNotIn("g3c4_p2g2_terminal_carriers_extension.esk", feature_off)

    def test_native_pre_sampling_failure_and_abort_order(self):
        owner = source("src/eshkol_transformer/g3c4_model_owner.c")
        reserve = section(
            owner, "void *et_g3c4_private_output_reserve_v1(",
            "int64_t et_g3c4_private_output_release_v1(",
        )
        self.assert_order(
            reserve, "et_g3c4_admit_active_call(context_candidate)",
            "et_g3c4_pending_output_lookup(context, &pending)", "if (pending != NULL)",
            "et_g3c4_output_allocate()", "et_i64_tensor_create_v1(",
            "et_g3c4_error_snapshot_internal()",
            "et_g3c4_output_cleanup_unpublished(output)",
            "et_g3c4_error_restore_internal(first)", "return NULL;",
            "output->transport.state = 0u;", "output->parent_ctx = context;",
            "et_g3c4_enroll_output(output)", "return output;",
        )
        self.assertNotRegex(reserve, r"context->\w+\s*=(?!=)")
        self.assertNotRegex(reserve, r"et_g3c4_\w*(?:prefill|sample|commit)\w*\(")
        abort = owner[owner.index("int64_t et_g3c4_private_call_abort_v1("):]
        self.assert_order(
            abort, "et_g3c4_output_discard_preflight(pending_output)",
            "et_a2_kv_cache_transaction_view_begin_v1(",
            "et_g3c4_token_frame_discard(context)",
            "et_g3c4_output_discard_pending(pending_output)",
        )


if __name__ == "__main__":
    unittest.main()
