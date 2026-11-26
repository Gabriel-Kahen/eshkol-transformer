"""Check the isolated draft fixture against both accepted source layouts."""

import re
import unittest

from tests.g3c4.test_p2_g2_constructor_reservation_boundary import source


FIXTURE = "tests/g3c4/p2_g2_pre_enrollment_draft_test.esk"


def no_comments(text):
    return re.sub(r";[^\n]*", "", text)


def binding_vector(text, binding):
    found = re.search(rf"\({binding} \(vector ([0-9 ]+)\)\)", text)
    return tuple(map(int, found.group(1).split()))


class PreEnrollmentDraftTests(unittest.TestCase):
    def test_project_original_objects_and_extents_in_pending_slot_order(self):
        pending = source("native/g3c4_p2g2_terminal_pending_anchor_extension.esk")
        constructor = source("native/g3c4_p2g2_terminal_carriers_extension.esk")
        fixture = source(FIXTURE)
        self.assertIn(
            "(anchor (vector call record (g3c4-p2g2-anchor-for call) "
            "output (vector-ref output 0) (vector-ref output 3) "
            "(vector-ref record 8) (vector-ref record 9) "
            "(vector-ref record 10) (vector-ref record 11) "
            "(vector-ref record 12) second-raw second-staging "
            "text-one text-two extents 'pending))",
            " ".join(pending.split()),
        )
        self.assertIn(
            "(bundle (vector first-raw first-staging pending-raw "
            "pending-staging second-raw second-staging text-one text-two extents 'live))",
            " ".join(constructor.split()),
        )
        order = binding_vector(fixture, "order")
        self.assertEqual(order, (2, 3, 0, 1, 4, 5, 6, 7))
        extents = binding_vector(constructor, "extents")
        self.assertEqual(tuple(extents[i] for i in order), binding_vector(pending, "extents"))
        self.assertIn("(vector-set! draft (+ 7 i) (vector-ref bundle carrier))", fixture)
        self.assertIn("(vector-set! extents i (vector-ref original-extents carrier))", fixture)

    def test_fixture_is_lexical_unbound_and_retires_its_bundle(self):
        fixture = source(FIXTURE)
        code = " ".join(no_comments(fixture).split())
        self.assertEqual(code.count("("), code.count(")"))
        self.assertEqual(re.findall(r'\(load "([^"]+)"\)', fixture), [
            "m3_package_root.esk",
            "g3c4_i2_construction_extension.esk",
            "g3c4_model_extension.esk",
            "g3c4_p2g2_terminal_carriers_extension.esk",
        ])
        self.assertIn("(let ((operation 'p2-g2-pre-enrollment-draft-test)) "
                      "(define (try-draft defect)", code)
        self.assertIn("(draft (make-vector 17 #f))", code)
        self.assertIn("(eq? (vector-ref candidate i) #f)", code)
        self.assertIn("(eq? (vector-ref candidate 16) 'unbound)", code)
        self.assertIn("(eq? (vector-ref candidate (+ 7 i)) (vector-ref bundle carrier))", code)
        self.assertIn("(g3c4-p2g2-terminal-carriers-retire! bundle) (raise caught)", code)
        self.assertIn("(g3c4-p2g2-terminal-carriers-retire! bundle) valid))", code)
        self.assertEqual(set(re.findall(r"\((g3c4-[\w!-]+)", code)), {
            "g3c4-p2g2-terminal-carriers-create",
            "g3c4-p2g2-terminal-carriers-check",
            "g3c4-p2g2-terminal-carriers-retire!",
        })
        self.assertNotRegex(code, r"\((?:extern|provide|export|eval|apply|set!)\s")
        self.assertNotIn(FIXTURE.rsplit("/", 1)[1], source("Makefile"))

    def test_negative_cases_are_exercised_between_two_fresh_drafts(self):
        code = " ".join(no_comments(source(FIXTURE)).split())
        defects = "false-owner equal-length-raw first-as-second extent published-state"
        self.assertIn(
            f"(check (try-draft 'authentic)) (let each ((defects '({defects}))) "
            "(if (pair? defects) (begin (check (not (try-draft (car defects)))) "
            "(each (cdr defects))))) (check (try-draft 'authentic))",
            code,
        )
        for defect in defects.split():
            with self.subTest(defect=defect):
                self.assertIn(f"(({defect}) (vector-set!", code)


if __name__ == "__main__":
    unittest.main()
