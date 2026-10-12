"""Reject lexical cleanup authority across the private pair guards."""
from pathlib import Path
import unittest

from tests.c2.test_handler_order import definition, parse, render


ROOT = Path(__file__).resolve().parents[2]
PAIR = ROOT / "native/g3r_file_pair_extension.esk"
BRIDGE = ROOT / "native/g3r_pair_rng_bridge_extension.esk"


def require(source: str, name: str, parts: tuple[str, ...]) -> str:
    body = render(definition({name: parse(source)}, name))
    if any(part not in body for part in parts):
        raise ValueError(f"{name} lost cell-backed cleanup authority")
    return body


def verify_pair(source: str) -> None:
    cleanup = require(source, "g3r-file-pair-cleanup!", (
        "(if (vector-ref borrow-cell 0) (g3r-c2-receipt-borrow-end-internal! (vector-ref borrow-cell 0)))",
        "(if (vector-ref digest-cell 0) (g3r-c2-sha-zero! (vector-ref digest-cell 0)))",
        "(if (vector-ref receipt-cell 0) (g3r-c2-receipt-release-internal! (vector-ref receipt-cell 0)))",
    ))
    if cleanup.index("g3r-c2-receipt-borrow-end-internal!") > cleanup.index(
            "g3r-c2-receipt-release-internal!"):
        raise ValueError("pair cleanup release order changed")
    admit = require(source, "g3r-file-pair-admit-internal", (
        "(let ((receipt-cell (vector #f)) (borrow-cell (vector #f)) (digest-cell (vector #f)))",
        "(g3r-file-pair-cleanup! receipt-cell borrow-cell digest-cell)",
        "(vector-set! receipt-cell 0 (g3r-c2-load-with-sha-internal",
        "(vector-set! borrow-cell 0 (g3r-c2-receipt-borrow-begin-internal",
        "(vector-set! digest-cell 0 (g3r-c2-receipt-borrow-sha-internal",
        "(vector-set! borrow-cell 0 #f)",
        "(g3r-c2-sha-zero! (vector-ref digest-cell 0)) (vector-set! digest-cell 0 #f)",
        "(vector-copy! destination 0 staged 0 9) (vector-set! receipt-cell 0 #f)",
    ))
    if "(set! receipt " in admit or "(set! borrow " in admit:
        raise ValueError("pair lexical owner reintroduced")


def verify_bridge(source: str) -> None:
    bridge = require(source, "g3r-pair-rng-admit-internal", (
        "(receipt-cell (vector #f)) (rng-owner-cell (vector #f))",
        "(if (vector-ref rng-owner-cell 0) (g3t-generation-rng-release! (vector-ref rng-owner-cell 0)))",
        "(if (or (vector-ref receipt-cell 0) (vector-ref pair 0)) (g3r-c2-receipt-release-internal!",
        "(vector-set! receipt-cell 0 (vector-ref pair 0))",
        "(let ((fresh-owner (g3r-rng-owner-from-checked-words",
        "(if (not (eq? (vector-ref rng-owner-cell 0) fresh-owner)) (g3t-generation-rng-release! fresh-owner))",
        "(vector-set! rng-owner-cell 0 fresh-owner)",
        "(vector-copy! destination 0 staged 0 10) (vector-set! pair 0 #f) (vector-set! rng-owner-cell 0 #f) (vector-set! receipt-cell 0 #f)",
    ))
    if "(set! receipt " in bridge or "(set! rng-owner " in bridge:
        raise ValueError("bridge lexical owner reintroduced")


class PairGuardAuthorityStaticTest(unittest.TestCase):
    def test_both_guards_keep_cell_authority(self):
        verify_pair(PAIR.read_text())
        verify_bridge(BRIDGE.read_text())

    def test_pair_lexical_cleanup_regression_rejected(self):
        source = PAIR.read_text().replace(
            "receipt-cell borrow-cell digest-cell))",
            "receipt borrow digest))", 1)
        self.assertNotEqual(source, PAIR.read_text())
        with self.assertRaisesRegex(ValueError, "cell-backed cleanup authority"):
            verify_pair(source)

    def test_bridge_lexical_cleanup_regression_rejected(self):
        source = BRIDGE.read_text().replace(
            "(g3t-generation-rng-release!\n                       (vector-ref rng-owner-cell 0))",
            "(g3t-generation-rng-release! rng-owner)", 1)
        self.assertNotEqual(source, BRIDGE.read_text())
        with self.assertRaisesRegex(ValueError, "cell-backed cleanup authority"):
            verify_bridge(source)


if __name__ == "__main__":
    unittest.main()
