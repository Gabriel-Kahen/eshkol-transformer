"""Source-only gate for the private terminal result prerequisite."""

from pathlib import Path
import runpy
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
OWNER = ROOT / "src/eshkol_transformer/g3c4_model_owner.c"
HEADER = ROOT / "src/eshkol_transformer/g3c4_context_internal.h"
TEST = ROOT / "tests/g3c4/p2_g2_public_result_owner_native.c"
BASE = "e9cd421258e164381c81da27e44336998e5df37b"
MACROS = runpy.run_path(str(ROOT / "scripts/check-g3c4-p2g2-terminal.py"))[
    "OWNER_MACROS"
] + [
    "ET_G3C4_P2_G2_TERMINAL_PRIVATE",
    "ET_G3C4_P2_G2_TERMINAL_IDS_CLONE_PRIVATE",
    "ET_A2_KV_CACHE_TERMINAL_WITNESS_PRIVATE",
    "ET_G3C4_T1_I1_EXACT_PAIR_PRIVATE",
    "ET_G3C4_P2_G1_PUBLICATION_PRIVATE",
    "ET_G3C4_P2_G1_COPYOUT_PRIVATE",
    "ET_G3C4_P2_G1_PUBLIC_RESULT_PRIVATE",
]
NEW = "ET_G3C4_P2_G2_PUBLIC_RESULT_PRIVATE"
INCLUDES = ["-I" + str(ROOT / path) for path in (
    "include", "native", "src", "src/eshkol_transformer")]


def cc(path, macros, mode):
    return subprocess.run(
        ["cc", "-std=c11", *mode, *INCLUDES,
         *("-D" + item for item in macros), str(path)],
        cwd=ROOT, capture_output=True, check=False)


class PrivateTerminalResultSource(unittest.TestCase):
    def test_feature_on_owner_and_native_negative_cases_are_c11(self):
        flags = ["-Wall", "-Wextra", "-Werror", "-Wpedantic", "-fsyntax-only"]
        for source in (OWNER, TEST):
            result = cc(source, MACROS + [NEW], flags)
            self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_feature_off_owner_and_header_are_byte_identical(self):
        with tempfile.TemporaryDirectory() as temporary:
            for relative, current in (
                ("src/eshkol_transformer/g3c4_model_owner.c", OWNER),
                ("src/eshkol_transformer/g3c4_context_internal.h", HEADER),
            ):
                old = Path(temporary) / Path(relative).name
                blob = subprocess.run(
                    ["git", "show", BASE + ":" + relative], cwd=ROOT,
                    capture_output=True, check=True).stdout
                old.write_bytes(blob)
                flags = ["-E", "-P", "-x", "c"]
                self.assertEqual(
                    cc(old, MACROS, flags).stdout,
                    cc(current, MACROS, flags).stdout,
                    relative + " changed with the new feature disabled")

    def test_new_symbol_has_exact_guard_and_feature_off_absence(self):
        symbol = b"et_g3c4_private_terminal_result_tensor_create_v1"
        flags = ["-E", "-P", "-x", "c"]
        for path in (OWNER, HEADER):
            self.assertNotIn(symbol, cc(path, MACROS, flags).stdout)
            self.assertIn(symbol, cc(path, MACROS + [NEW], flags).stdout)
        for macros in ([NEW], [NEW, "ET_G3C4_P2_G1_PUBLIC_RESULT_PRIVATE"]):
            result = cc(OWNER, macros, ["-E", "-x", "c"])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"requires the scalar result and terminal clone",
                          result.stderr)


if __name__ == "__main__":
    unittest.main()
