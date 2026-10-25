"""Keep the documented workspace blocker tied to the current P1 authority."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
P1 = (ROOT / "internal/p1/lib/transformer/module.esk").read_text()
ASSOCIATION = (
    ROOT / "native/g3c4_trainable_owner_association_extension.esk"
).read_text()


def body(source: str, start: str, end: str) -> str:
    return source.split(start, 1)[1].split(end, 1)[0]


class WorkspaceOwnershipBlockerSource(unittest.TestCase):
    def test_slot74_checks_train_at_capture_and_recheck(self) -> None:
        mode_gate = "((not (eq? (vector-ref node 1) e3-mode-train)) 2)"
        capture = body(P1, "(define (tr3-fixed-capture-module!", "(define (tr3-fixed-capture-modules!")
        recheck = body(P1, "(define (tr3-fixed-recheck-module ", "(define (tr3-fixed-recheck-modules ")
        self.assertIn(mode_gate, capture)
        self.assertIn(mode_gate, recheck)

    def test_association_requires_empty_workspace_edge(self) -> None:
        self.assertIn("(eq? (vector-ref entry 10) #f)", ASSOCIATION)
        self.assertEqual(ASSOCIATION.count("(module-c4-train-topology-status-internal"), 2)

    def test_existing_identity_read_returns_only_carrier(self) -> None:
        getter = body(
            P1,
            "(define (parameter-handle-tensor-identity-internal handle)",
            "(define ",
        )
        self.assertIn("(vector-ref (require-handle 'parameter-handle-tensor-identity-internal handle) 6)", getter)


if __name__ == "__main__":
    unittest.main()
