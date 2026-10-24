"""Negative controls for the finite local-transfer verifier."""

import hashlib
import unittest

from tests.cpuv1_resource.verify_local_transfer import HEADERS, ROWS, check_rows


def fixture():
    rows = {
        line: (instruction, hashlib.sha256(instruction.encode()).hexdigest())
        for selected in ROWS.values()
        for line, instruction in selected.items()
    }
    functions = [
        {"name": name, "header": f"define @x({capture})"}
        for name, capture in HEADERS.items()
    ]
    return rows, functions


class LocalTransferNegatives(unittest.TestCase):
    def test_expected_selected_rows(self):
        rows, functions = fixture()
        errors, evidence = check_rows(rows, functions)
        self.assertEqual(errors, [])
        self.assertEqual([len(evidence[k]) for k in ("p1", "candidate")], [15, 15])

    def test_replaced_holder_pointer_is_rejected(self):
        rows, functions = fixture()
        text, digest = rows[25214989]
        rows[25214989] = (text.replace("%8990094", "%8990114"), digest)
        errors, _ = check_rows(rows, functions)
        self.assertIn("p1: missing or changed MAIN instruction at line 25214989", errors)

    def test_changed_capture_offset_is_rejected(self):
        rows, functions = fixture()
        text, digest = rows[31430270]
        rows[31430270] = (text.replace("i64 24", "i64 40"), digest)
        errors, _ = check_rows(rows, functions)
        self.assertIn("candidate: missing or changed MAIN instruction at line 31430270", errors)

    def test_missing_binding_or_typed_capture_is_rejected(self):
        rows, functions = fixture()
        del rows[25168108]
        functions[1]["header"] = "define @x(ptr %captured_other)"
        errors, _ = check_rows(rows, functions)
        self.assertEqual(len(errors), 2)


if __name__ == "__main__":
    unittest.main()
