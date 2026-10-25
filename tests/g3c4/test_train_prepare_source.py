#!/usr/bin/env python3
"""Source and mutation checks for the private C4 train-prepared sibling."""

from hashlib import sha256
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


def definition(source: str, name: str) -> str:
    start = source.index(f"(define ({name}")
    depth = 0
    quoted = False
    escaped = False
    comment = False
    for index in range(start, len(source)):
        char = source[index]
        if comment:
            comment = char != "\n"
        elif quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == ";":
            comment = True
        elif char == '"':
            quoted = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise AssertionError(f"unbalanced source form: {name}")


def check(p1: str, i2: str, wrapper: str, fixture: str) -> None:
    eval_prepare = definition(p1, "construction-prepare-eval!")
    train_prepare = definition(p1, "construction-prepare-train!")
    assert sha256(eval_prepare.encode()).hexdigest() == (
        "3915874e76d84bca0a3a13421ee077163d0981349df32c9eb500ac3699255f7b"
    )
    assert train_prepare == eval_prepare.replace(
        "construction-prepare-eval!", "construction-prepare-train!"
    ).replace(
        "module-construction-prepare-eval-internal!",
        "module-construction-prepare-train-internal!",
    ).replace("'eval", "'train")
    eval_guard = definition(p1, "construction-prepare-eval-guarded!")
    train_guard = definition(p1, "construction-prepare-train-guarded!")
    assert sha256(eval_guard.encode()).hexdigest() == (
        "320798f98241552fd2bb7f689af7ab48f026fcb37b6900bf6f0716cc00691098"
    )
    assert train_guard == eval_guard.replace(
        "construction-prepare-eval-guarded!", "construction-prepare-train-guarded!"
    ).replace(
        "construction-prepare-eval!", "construction-prepare-train!"
    ).replace(
        "module-construction-prepare-eval-internal!",
        "module-construction-prepare-train-internal!",
    )
    assert p1.count("(construction-prepare-train-guarded! identity))") == 1
    assert "(eq? operation 'module-construction-prepare-train-internal!)" in p1
    assert wrapper.count("(vector-ref p1-trusted-surface 73)") == 1
    assert wrapper.count("(vector-ref p1-trusted-surface 71)") == 1
    assert wrapper.count("(vector-ref p1-trusted-surface 72)") == 1
    route = definition(i2, "i2-g3c4-construction-prepare-train-internal!")
    for marker in (
        "(i2-construction-open-route value 'g3c4 operation)",
        "(i2-construction-retained-preflight! record operation)",
        "(i2-construction-rollback-tail! record)",
        "(module-construction-prepare-train-internal!",
        "(vector-set! record 3 'prepared)",
    ):
        assert marker in route
    assert route.index("(module-construction-prepare-train-internal!") < route.index(
        "(vector-set! record 3 'prepared)"
    )
    for marker in (
        "missing schedule aborts complete I2 route",
        "failed prepare removes I2 root",
        "prepared abort",
        "eval sibling cannot prepare twice",
        "canonical root in train mode",
        "native owner commit",
    ):
        assert marker in fixture


class TrainPrepareSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sources = (
            (ROOT / "internal/p1/lib/transformer/module.esk").read_text(),
            (ROOT / "native/g3c4_i2_construction_extension.esk").read_text(),
            (ROOT / "native/g3c4_p1_construction_extension.esk").read_text(),
            (ROOT / "tests/g3c4/i2_train_prepared_route_test.esk").read_text(),
        )

    def test_exact_sibling_and_controls(self) -> None:
        check(*self.sources)
        template = (ROOT / "templates/p1/module_roots.esk.tmpl").read_text()
        trusted = template.split("@@P1_TRUSTED_BEGIN@@\n", 1)[1].split(
            "@@P1_TRUSTED_END@@", 1
        )[0]
        self.assertEqual(trusted, self.sources[0])

    def test_mutated_train_mode_rejected(self) -> None:
        changed = list(self.sources)
        changed[0] = changed[0].replace(
            "(set-mode-modules! canonical-finalizers 'train)",
            "(set-mode-modules! canonical-finalizers 'eval)", 1
        )
        with self.assertRaises(AssertionError):
            check(*changed)

    def test_missing_i2_rollback_rejected(self) -> None:
        changed = list(self.sources)
        changed[1] = changed[1].replace(
            ";; The P1 train prepare guard revokes its admitted ledger first.\n"
            "               (i2-construction-rollback-tail! record)",
            ";; missing rollback", 1
        )
        with self.assertRaises(AssertionError):
            check(*changed)

    def test_wrong_slot_rejected(self) -> None:
        changed = list(self.sources)
        changed[2] = changed[2].replace("p1-trusted-surface 73", "p1-trusted-surface 71")
        with self.assertRaises(AssertionError):
            check(*changed)

    def test_slot73_test_overlays_are_old_source_pinned(self) -> None:
        old_hash = "8629d7ee347fb81d0a24cbb5d296828208af9bf75965244d70f7a05ecb132e40"
        current_hash = sha256(self.sources[0].encode()).hexdigest()
        self.assertNotEqual(old_hash, current_hash)
        e3_overlay = (ROOT / "scripts/generate-e3-zero-p1-topology-overlay.py").read_text()
        schedule = (ROOT / "scripts/test-p1-construction-schedule.sh").read_text()
        registry = (ROOT / "scripts/check-p1-registry-publication.sh").read_text()
        self.assertIn(f'SOURCE_SHA256 = "{old_hash}"', e3_overlay)
        self.assertIn("Test-only probes; the production trusted surface is unchanged", schedule)
        self.assertIn("Test-only slot 73 models failed advisory linkage", registry)


if __name__ == "__main__":
    unittest.main()
