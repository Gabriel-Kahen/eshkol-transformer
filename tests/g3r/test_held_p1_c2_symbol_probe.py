"""Regression for the held P1/C2 allocator-symbol evidence gate."""

from pathlib import Path
import os
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "scripts/check-g3r-f32-allocator-symbol.sh"
SYMBOL = "et_f32_tensor_test_fail_alloc_after_v1"


class AllocatorSymbolProbeTest(unittest.TestCase):
    def test_defined_text_symbol_and_missing_symbol(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            filler = "\n".join(
                f"int zzzz_probe_padding_{index:03d}(void) {{ return {index}; }}"
                for index in range(250)
            )
            for expected in (True, False):
                stem = "present" if expected else "missing"
                source = work / f"{stem}.c"
                obj = work / f"{stem}.o"
                evidence = work / f"{stem}.symbols"
                source.write_text(
                    (f"void {SYMBOL}(void) {{}}\n" if expected else "")
                    + filler + "\n"
                )
                subprocess.run(
                    [os.environ.get("CC", "cc"), "-std=c11", "-c", str(source),
                     "-o", str(obj)], check=True
                )
                result = subprocess.run(
                    ["bash", str(PROBE), str(obj), str(evidence)],
                    capture_output=True, text=True
                )
                self.assertEqual(result.returncode == 0, expected, result.stderr)
                self.assertEqual(
                    evidence.read_bytes(),
                    subprocess.check_output(
                        ["nm", "-g", "--defined-only", "--format=posix", str(obj)],
                        env={**os.environ, "LC_ALL": "C"}
                    )
                )
                self.assertGreater(evidence.stat().st_size, 4096)
                self.assertEqual(SYMBOL in evidence.read_text(), expected)


if __name__ == "__main__":
    unittest.main()
