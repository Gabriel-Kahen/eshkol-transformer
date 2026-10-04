"""No toolchain needed: private overlay and ordinary-source isolation checks."""

from pathlib import Path
import hashlib
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "scripts/build-g3r-candidate-p1-overlay.py"
MODULE = ROOT / "internal/p1/lib/transformer/module.esk"
PRIVATE = ROOT / "native/g3r_candidate_construction_extension.esk"
ROOT_SOURCE = ROOT / "native/g3r_candidate_construction_root.esk"
ORIGINAL_MODULE_SHA = "1146eee9e053b4a761c24f56b75ed43e756ec2d907d0de48b04e72abe805775d"
UNCHANGED = {
    "native/m3t_package_root.esk": "f3f804cedff58b87c0ba08d3a30a3559bd2d0850d53558a7b3a21b4fd7e01c98",
    "native/i2_wave2_root.esk": "f1314ccd4c5a382733115005c93d8712290bd63c5373eac2509093989fefad33",
    "native/m3t_transport_extension.esk": "71247e722d72764ce1b3400cb3f818ba12c3754dd11a6c742ffcec63942d5815",
    "native/p1_identity.c": "155b2fe2fb6b26c6cd089f56390659cf3790265263e0d34f76b8dc06c8cae90f",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CandidateConstructionStatic(unittest.TestCase):
    def test_ordinary_sources_are_exactly_unchanged(self) -> None:
        self.assertEqual(digest(MODULE), ORIGINAL_MODULE_SHA)
        for relative, expected in UNCHANGED.items():
            with self.subTest(relative=relative):
                self.assertEqual(digest(ROOT / relative), expected)
        self.assertNotIn("candidate-construction-begin", MODULE.read_text())

    def test_private_overlay_is_deterministic_and_noninstalled(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            for destination in (first, second):
                subprocess.run(
                    ["python3", str(GENERATOR), destination],
                    cwd=ROOT, check=True, capture_output=True, text=True,
                )
            left = Path(first) / "transformer/module.esk"
            right = Path(second) / "transformer/module.esk"
            self.assertEqual(left.read_bytes(), right.read_bytes())
            source = left.read_text()
            self.assertEqual(source.count("et_p1_private_candidate_construction_begin_v1"), 1)
            self.assertEqual(source.count("et_p1_private_candidate_graph_preflight_v1"), 1)
            self.assertIn("(vector-ref p1-trusted-surface 73)", source)
            self.assertIn("(vector-ref p1-trusted-surface 74)", source)
            for slot in range(75, 80):
                self.assertIn(f"(vector-ref p1-trusted-surface {slot})", source)
            self.assertIn("(define g3r-candidate-ledgers (vector '() #f))", source)
            self.assertEqual(source.count("(define p1-trusted-surface"), 1)
            self.assertLess(source.index("(define g3r-candidate-ledgers"),
                            source.index("(vector\n  (lambda (module)"))
            self.assertNotIn("(define g3r-candidate-ledgers", source[source.index(";; Candidate-only wrapper"):])
            self.assertIn("(candidate-construction-shells identity)", source)
            self.assertIn("(if irreversible? (exit 134)", source)
        self.assertEqual(digest(MODULE), ORIGINAL_MODULE_SHA)

    def test_closed_private_root_and_enrollment_order(self) -> None:
        root = ROOT_SOURCE.read_text()
        self.assertEqual(
            [line for line in root.splitlines() if line.startswith("(load ")],
            ['(load "m3t_package_root.esk")'],
        )
        source = PRIVATE.read_text()
        self.assertNotIn("(provide ", source)
        for name in (
            "g3r-candidate-ledger-begin-internal!",
            "g3r-candidate-model-create-internal!",
            "i2-candidate-construction-begin-internal!",
        ):
            self.assertIn(name, source)
        self.assertIn("(define (candidate-construction-begin",
                      (ROOT / "native/g3r_candidate_p1_inside.esk").read_text())
        self.assertLess(source.index("(vector-set! g3r-candidate-ledgers 0 next)"),
                        source.index("(define (g3r-candidate-model-create-internal!"))
        self.assertIn("(eq? (vector-ref row 7) 'armed)",
                      (ROOT / "native/g3r_candidate_p1_inside.esk").read_text())
        self.assertLess((ROOT / "native/g3r_candidate_p1_inside.esk").read_text().index(
                        "(vector-set! row 7 'consumed)"),
                        (ROOT / "native/g3r_candidate_p1_inside.esk").read_text().index(
                        "(p1-native-candidate-construction-begin native-context)"))
        self.assertLess(source.index("(set! native (m3t-native-owner-create"),
                        source.index("(vector-set! (vector-ref canonical 4) 0 native)"))
        self.assertEqual(set(re.findall(r"\(vector-set! row (\d+)", source)),
                         {"7", "8"})
        self.assertIn("(vector-set! row 7 'armed)\n"
                      "          (let ((construction\n"
                      "                  (candidate-construction-begin\n"
                      "                    i2-provider-name p1-cell)))", source)
        self.assertLess(source.index("(vector-set! enrollment-cell 3 (vector-ref record 4))"),
                        source.index("(i2-construction-seal-internal! record)"))
        self.assertLess(source.index("(g3r-candidate-test-cut 4)"),
                        source.index("(i2-candidate-construction-seal-internal!\n"))


if __name__ == "__main__":
    unittest.main()
