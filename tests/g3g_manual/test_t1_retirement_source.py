"""Source-only isolation checks for the private T1 retirement fixture."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/g3g_manual/private_t1_retirement_runtime.esk"
SYMBOL = "et_t1_i64_shell_test_release_sealed_v1"


def balanced_eshkol(source: str) -> bool:
    depth = 0
    quoted = False
    escaped = False
    commented = False
    for char in source:
        if commented:
            if char == "\n":
                commented = False
        elif quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == ";":
            commented = True
        elif char == '"':
            quoted = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0 and not quoted


class RetirementSourceTest(unittest.TestCase):
    def test_private_fixture_has_one_package_root(self):
        source = FIXTURE.read_text()
        self.assertTrue(balanced_eshkol(source))
        self.assertEqual(
            re.findall(r'\(load\s+"([^"]+)"\)', source),
            ["g3g_manual_package_root.esk"],
        )

    def test_test_symbol_is_not_installed(self):
        self.assertIn(SYMBOL, (ROOT / "native/t1_i64_shell.c").read_text())
        self.assertIn(SYMBOL, FIXTURE.read_text())
        for relative in (
            "native/g3g_manual_package_root.esk",
            "native/g3g_manual_package_defined_symbols.txt",
            "native/g3g_manual_package_public_exports.txt",
            "tests/g3g_manual/public_g2_runtime.esk",
        ):
            path = ROOT / relative
            if path.exists():
                self.assertNotIn(SYMBOL, path.read_text(), relative)


if __name__ == "__main__":
    unittest.main()
