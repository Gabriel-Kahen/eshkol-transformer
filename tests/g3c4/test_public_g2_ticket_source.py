"""Owned source controls for the prezero public G2 call ticket."""

from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
check = SourceFileLoader(
    "public_g2_ticket_check", str(ROOT / "scripts/check-g3c4-public-g2-ticket.py")
).load_module()


class TicketSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        read = lambda path: (ROOT / path).read_text()
        cls.inputs = dict(
            dispatch=read("native/g3c4_public_generation_dispatch_extension.esk"),
            terminal=read("native/g3c4_p2g2_terminal_close_extension.esk"),
            call_entry=read("native/g3c4_call_entry_extension.esk"),
            adapter=(ROOT / "native/g3c4_public_output_adapter_extension.esk").read_bytes(),
            fixture=read("tests/g3c4/p2_g1_public_generation_test.esk"),
            g2_close_fixture=read("tests/g3c4/p2_g2_public_ticket_close_test.esk"),
            package_root=read("native/g3g_manual_package_root.esk"),
            package_sources=read("native/g3g_manual_package_source_closure.txt").splitlines(),
            prefix_negative=read("tests/g3c4/p2_g1_g2_prefix_stub_negative_test.esk"),
            core_negative=read("tests/g3c4/p2_g1_g2_core_stub_negative_test.esk"),
            closure=check.CLOSURE.read_text().splitlines(),
            base=check.BASE.read_text().splitlines(),
        )

    def reject(self, **changes: object) -> None:
        candidate = dict(self.inputs)
        candidate.update(changes)
        with self.assertRaises(AssertionError):
            check.validate(**candidate)

    def test_original_shape(self) -> None:
        check.validate(**self.inputs)

    def test_global_issuer_rejected(self) -> None:
        self.reject(dispatch=self.inputs["dispatch"] +
                    "\n(define g3c4-public-g2-bridge (lambda (x) x))\n")

    def test_public_callable_must_issue(self) -> None:
        self.reject(dispatch=self.inputs["dispatch"].replace(
            "'generate generator input #f #f #f #f 0 -1 -1",
            "'admit generator input #f #f #f #f 0 -1 -1",
            1))

    def test_public_callable_has_library_init_storage(self) -> None:
        self.reject(dispatch=self.inputs["dispatch"].replace(
            "(define g3c4-public-generator-generate\n"
            "  (lambda (generator input) (exit 134)))",
            "(define (g3c4-public-generator-generate generator input)\n"
            "  (exit 134))", 1))

    def test_budget_one_cannot_reach_ticket(self) -> None:
        self.reject(dispatch=self.inputs["dispatch"].replace(
            "(if (= (vector-ref (vector-ref generator-entry 7) 5) 2)",
            "(if (= (vector-ref (vector-ref generator-entry 7) 5) 1)", 1))

    def test_dead_generator_close_preserves_idempotence(self) -> None:
        self.reject(dispatch=self.inputs["dispatch"].replace(
            "(if (eq? (vector-ref member 2) 'dead)", "(if #f", 1))
        self.reject(fixture=self.inputs["fixture"].replace(
            "(check \"authentic dead P2/G1 generator close stays idempotent\"\n"
            "  (g3c4-public-generator-close greedy))", "", 1))
        self.reject(call_entry=self.inputs["call_entry"].replace(
            "((g3c4-dead-entry? entry 'generator) #t)",
            "((g3c4-dead-entry? entry 'generator)\n"
            " (g3c4-native-generator-close (vector-ref entry 3)))", 1))
        self.reject(g2_close_fixture=self.inputs["g2_close_fixture"].replace(
            "authentic dead public G2 generator close stays idempotent",
            "missing repeated close", 1))
        self.reject(g2_close_fixture=self.inputs["g2_close_fixture"].replace(
            "malformed dead public G2 generator rejects",
            "missing malformed denial", 1))

    def test_creation_time_profile_required(self) -> None:
        self.reject(dispatch=self.inputs["dispatch"].replace(
            "(g3c4-public-g2-generator-create\n            model",
            "(g3c4-generator-create-with-source\n            model", 1))
        self.reject(dispatch=self.inputs["dispatch"].replace(
            "(eq? (vector-ref row 6) 'diagnostic-c4-p2g2)", "#t", 1))

    def test_secret_and_phase_required(self) -> None:
        self.reject(dispatch=self.inputs["dispatch"].replace(
            "(eq? (vector-ref ticket 0) secret)", "#t", 1))
        self.reject(dispatch=self.inputs["dispatch"].replace(
            "(eq? (vector-ref ticket 12) 'pending)", "#t", 1))

    def test_binding_key_checked_before_zero(self) -> None:
        self.reject(dispatch=self.inputs["dispatch"].replace(
            "g3c4-public-g2-output-bindings 0))\n                                    0)",
            "g3c4-public-g2-output-bindings 0))\n                                    1)", 1))

    def test_precommit_verify_required(self) -> None:
        self.reject(terminal=self.inputs["terminal"].replace(
            "(public-stage 'verify output row anchor count selected-first",
            "(vector 'verify output row anchor count selected-first", 1))

    def test_old_adapter_and_load_order(self) -> None:
        self.reject(adapter=self.inputs["adapter"] + b"\n;; drift\n")
        self.reject(package_root=self.inputs["package_root"].replace(
            '(load "g3c4_p2g2_terminal_close_extension.esk")\n', "", 1))

    def test_stub_closure_and_direct_negatives(self) -> None:
        self.reject(fixture=self.inputs["fixture"].replace(
            "(define (g3c4-p2g2-prefix-preflight generator input)\n  (exit 134))",
            "(define (g3c4-p2g2-prefix-preflight generator input)\n  #t)", 1))
        self.reject(core_negative=self.inputs["core_negative"].replace(
            "g3c4-p2g2-terminal-close-core-private", "missing-core", 1))

    def test_closure_member_missing(self) -> None:
        self.reject(closure=self.inputs["closure"][:-1])
        self.reject(package_sources=self.inputs["package_sources"][:-1])


if __name__ == "__main__":
    unittest.main()
