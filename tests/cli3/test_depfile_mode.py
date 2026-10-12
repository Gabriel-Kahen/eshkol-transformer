"""Keep CLI3 depfile checks on Eshkol's supported object output mode."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class DepfileModeTests(unittest.TestCase):
    def test_standalone_gate_checks_object_depfile_before_executable(self) -> None:
        source = (ROOT / "scripts" / "test-cli3-corpus-validate.sh").read_text()
        object_compile = 'cli3_run --emit-object --emit-depfile "${output}.d"'
        depfile_check = '[[ -s "${output}.d" ]]'
        executable_compile = 'cli3_run "${source}" -o "${output}"'
        self.assertLess(source.index(object_compile), source.index(depfile_check))
        self.assertLess(source.index(depfile_check), source.index(executable_compile))
        self.assertIn('cli3_step compile-command cli3_compile', source)
        self.assertIn('cli3_step compile-corpus-writer cli3_compile', source)
        self.assertIn('cli3_step compile-newline-writer cli3_compile', source)


if __name__ == "__main__":
    unittest.main()
