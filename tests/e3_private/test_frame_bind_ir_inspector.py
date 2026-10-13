"""Offline controls for the E3 canonical frame-bind IR pattern gate."""

from __future__ import annotations

import unittest

from tests.e3_private.inspect_frame_bind_ir import inspect


GOOD = b'''define linkonce_odr %eshkol_tagged_value @"e3-frame-bind!__eshkol_internal_abi"(%eshkol_tagged_value %model, %eshkol_tagged_value %destinations) sanitize_address {
entry:
  %canonical = alloca %eshkol_tagged_value, align 8
  %before = call %eshkol_tagged_value @consume(%eshkol_tagged_value %model)
  %canonical.load = load %eshkol_tagged_value, ptr %canonical, align 8
  %slot_check = icmp sge i64 2, %length
  %vref_result1 = phi %eshkol_tagged_value [ %root_value, %root_branch ]
  %root-model = alloca %eshkol_tagged_value, align 8
  store %eshkol_tagged_value %vref_result1, ptr %root-model, align 8
  %root-setup-box = alloca %eshkol_tagged_value, align 8
  %root-model.load = load %eshkol_tagged_value, ptr %root-model, align 8
  %root-setup-box.load = load %eshkol_tagged_value, ptr %root-setup-box, align 8
  %call = call %eshkol_tagged_value @"p1-e3-mode-bind!__eshkol_internal_abi"(%eshkol_tagged_value %root-model.load, %eshkol_tagged_value %root-setup-box.load)
  ret %eshkol_tagged_value %call
}
'''


class FrameBindIRInspectorTests(unittest.TestCase):
    def test_ssa_parameter_and_initial_stack_store_are_accepted(self) -> None:
        self.assertEqual(inspect(GOOD)["old_model_storage"], "ssa_parameter")
        stack = GOOD.replace(
            b"  %canonical = alloca",
            b"  %model1 = alloca %eshkol_tagged_value, align 8\n"
            b"  store %eshkol_tagged_value %model, ptr %model1, align 8\n"
            b"  %canonical = alloca",
        )
        self.assertEqual(inspect(stack)["old_model_storage"], "stack_slot")

    def test_rejects_old_binding_after_readback(self) -> None:
        for old_slot in (False, True):
            body = GOOD
            if old_slot:
                body = body.replace(
                    b"  %canonical = alloca",
                    b"  %model1 = alloca %eshkol_tagged_value, align 8\n"
                    b"  store %eshkol_tagged_value %model, ptr %model1, align 8\n"
                    b"  %canonical = alloca",
                )
            late_use = (b"  %late = load %eshkol_tagged_value, ptr %model1, align 8\n"
                        if old_slot else
                        b"  %late = call %eshkol_tagged_value @consume(%eshkol_tagged_value %model)\n")
            with self.subTest(old_slot=old_slot), self.assertRaisesRegex(
                ValueError, "old model binding is used after root readback"
            ):
                inspect(body.replace(b"  %root-model.load =", late_use + b"  %root-model.load ="))

    def test_rejects_reassignment_and_borrow_checks(self) -> None:
        mutants = (
            (GOOD.replace(b"  %root-model = alloca",
                          b"  %model.borrow_check = load %eshkol_tagged_value, ptr %model1, align 8\n"
                          b"  %root-model = alloca"), "old model binding still has a borrow check"),
            (GOOD.replace(b"  %root-model = alloca",
                          b"  store %eshkol_tagged_value %vref_result1, ptr %model1, align 8\n"
                          b"  %root-model = alloca"), "unrecognized storage shape"),
            (GOOD.replace(b"  %canonical = alloca",
                          b"  %model1 = alloca %eshkol_tagged_value, align 8\n"
                          b"  store %eshkol_tagged_value %model, ptr %model1, align 8\n"
                          b"  store %eshkol_tagged_value %model, ptr %model1, align 8\n"
                          b"  %canonical = alloca"), "old model binding was reassigned"),
            (GOOD.replace(b"%eshkol_tagged_value %root-model.load, %eshkol_tagged_value %root-setup-box.load",
                          b"%eshkol_tagged_value %other-model, %eshkol_tagged_value %root-setup-box.load"),
             "P1 bind does not consume"),
        )
        for body, message in mutants:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                inspect(body)


if __name__ == "__main__":
    unittest.main()
