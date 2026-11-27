"""Pin the private first-enrollment witness and its feature boundary."""

from pathlib import Path
import re
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]
EXTENSION = "native/g3c4_p2g2_first_enrollment_extension.esk"
FIXTURE = "tests/g3c4/p2_g2_first_enrollment_test.esk"
REFERENCES = (
    r"p2[-_]?g2[-_]first[-_]enrollment|g3c4[-_]first[-_]enrollment"
    r"|g3c4-p2g2-shell-count|abort-only-first-enrollment-witness"
)


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def scheme(path):
    return re.sub(r";[^\n]*|\"[^\"\n]*\"", "", read(path))


class FirstEnrollmentTests(unittest.TestCase):
    def assert_order(self, body, *parts):
        offset = 0
        for part in parts:
            offset = body.find(part, offset)
            self.assertGreaterEqual(offset, 0, part)
            offset += len(part)

    def test_original_pair_precedes_the_only_registry_enrollment(self):
        code = scheme(EXTENSION)
        for path in (EXTENSION, FIXTURE):
            depth = 0
            for char in scheme(path):
                depth += (char == "(") - (char == ")")
                self.assertGreaterEqual(depth, 0, path)
            self.assertEqual(depth, 0, path)
        self.assert_order(
            code,
            "(bundle (g3c4-p2g2-terminal-carriers-create))",
            "(pending-raw (vector-ref bundle 2))",
            "(pending-staging (vector-ref bundle 3))",
            "(auxiliary (vector pending-raw pending-staging))",
            "tokenizer #f auxiliary #f #f g3c4-model-profile",
            "(call-node (cons call previous-registry))",
            "(output-node (cons entry call-node))",
            "(g3c4-native-call-acquire native 2 2)",
            "(vector-set! g3c4-registry 0 output-node)",
            "(verify-roots)",
            "(g3c4-native-output-reserve native 2)",
        )
        self.assertEqual(code.count("(vector-set! g3c4-registry 0 output-node)"), 1)
        self.assertNotRegex(code, r"\(vector-set!\s+(entry\s+8|auxiliary\s+[01])\b")
        self.assertIn("(= (g3c4-p2g2-shell-count shell) 1)", code)
        self.assertIn("(scan (cdr entries)", code)

    def test_only_acquire_and_reservation_are_called_directly(self):
        code = scheme(EXTENSION)
        calls = set(re.findall(r"\((g3c4-native-[\w!-]+)", code))
        self.assertEqual(calls, {
            "g3c4-native-check", "g3c4-native-call-acquire",
            "g3c4-native-output-reserve", "g3c4-native-fail",
        })
        self.assert_order(
            code, "(g3c4-p2g2-protected-abort! call)",
            "(vector-set! state 0 'drained)",
            "(vector-set! g3c4-registry 0 previous-registry)",
            "(vector-set! g3c4-p2g2-first-enrollment-roots 0 previous-roots)",
            "(g3c4-p2g2-terminal-carriers-retire! bundle)",
            "(vector-set! state 0 'closed)",
        )
        self.assertIn("((drained) (exit 134))", code)
        self.assertNotRegex(code, r"\((extern|export|provide)\s")

    def test_references_are_confined_to_the_private_witness(self):
        # Check source, build and installation manifests, and workflows against
        # the same exact path set. Unrelated stack changes do not affect it.
        result = subprocess.run(
            ["git", "grep", "--untracked", "-IilzE", REFERENCES, "--", "."],
            check=True, capture_output=True, text=True, cwd=ROOT,
        )
        self.assertEqual(set(result.stdout.split("\0")[:-1]), {
            EXTENSION,
            FIXTURE,
            "tests/g3c4/p2_g2_first_enrollment_native.c",
            "tests/g3c4/test_p2_g2_first_enrollment.py",
            "scripts/test-g3c4-p2g2-first-enrollment.sh",
            "docs/ROADMAP.md",
            "docs/g3/G3_C4_P2_G2_ESHKOL_TERMINAL_CLOSE_CONTRACT.md",
        })
        leaf = Path(EXTENSION).name
        self.assertEqual(read(FIXTURE).count(f'(load "{leaf}")'), 1)


if __name__ == "__main__":
    unittest.main()
