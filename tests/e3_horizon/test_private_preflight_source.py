"""Source prerequisite for the still-pending compiled private 8,193 gate."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class PrivatePreflightSourceTests(unittest.TestCase):
    def test_d2_core_remaining_rows_guard_precedes_prepared_publication(self) -> None:
        source = (ROOT / "internal/e3/lib/e3_d2_restore.esk").read_text()
        body = source.split("(define (e3-d2-prepare! frame dataset)", 1)[1]
        body = body.split("(define (e3-d2-restore-preflight! frame)", 1)[0]
        total = "(vector-ref core 14)"
        next_row = "(vector-ref core 15)"
        remaining = f"(let ((remaining (- {total} {next_row})))"
        rejection = '(et-e1b-private-raise \'unsupported \'e3-evaluate-into!'
        publication = "(vector-set! record 3 'prepared)"
        self.assertIn("(= (vector-length core) 17)", body)
        self.assertIn(f"(<= {next_row} {total})", body)
        self.assertIn(remaining, body)
        self.assertIn("(> remaining 8192)", body)
        self.assertIn("(> (* remaining 2) 16384)", body)
        self.assertLess(body.index(remaining), body.index(rejection,
                            body.index(remaining)))
        self.assertLess(body.index(rejection, body.index(remaining)),
                        body.index(publication))

    def test_stage_two_precedes_native_acquire_and_traversal(self) -> None:
        source = (ROOT / "native/e3_private_extension.esk").read_text()
        body = source.split("(define (e3-evaluate-closed! model dataset frame entry inspect-args)", 1)[1]
        body = body.split("(define (e3-evaluate-into! model dataset frame)", 1)[0]
        stage = "(e3-stage-set! entry 2)"
        prepare = "(e3-d2-prepare! frame dataset)"
        acquire = "(e3-native-acquire (vector-ref entry 4))"
        traversal = "(e3-traverse! frame dataset)"
        for marker in (stage, prepare, acquire, traversal):
            self.assertIn(marker, body)
        self.assertLess(body.index(stage), body.index(prepare))
        self.assertLess(body.index(prepare), body.index(acquire))
        self.assertLess(body.index(acquire), body.index(traversal))


if __name__ == "__main__":
    unittest.main()
