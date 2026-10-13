"""Source admission for the pending compiled held-holder gate."""
from pathlib import Path
import unittest

from tests.c2.test_handler_order import parse


ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "tests/g3c4/p2_g2_held_holder_compiled_test.esk"
SCRIPT = ROOT / "scripts/test-g3c4-p2g2-held-holder-compiled.sh"
SHIM = ROOT / "tests/g3c4/p2_g2_held_holder_compiled_native.c"


class HeldHolderCompiledSource(unittest.TestCase):
    def test_driver_parses_and_reaches_exact_private_entry(self):
        source = DRIVER.read_text()
        self.assertTrue(parse(source))
        self.assertIn('(load "g3c4_p2g2_terminal_held_holder_root.esk")', source)
        self.assertIn("(g3c4-p2g2-terminal-generate-held-internal", source)
        self.assertIn("(normal-case 1 2 first-id 1)", source)
        self.assertIn("(normal-case 2 2 first-id 1)", source)
        self.assertIn("(p3-first-eos-reject first-id)", source)
        for kind, phase in ((1, 2), (2, 2), (1, 3), (2, 3)):
            self.assertIn(f"(retry-case {kind} {phase})", source)

    def test_fault_cuts_check_retry_and_original_roots(self):
        source = DRIVER.read_text()
        for fragment in (
            "wrong generation rejected before transfer",
            "exact first typed busy and one end/retry",
            "native context rolled back",
            "original ledgers and holder root restored",
            "holder disposed before terminal zero",
            '"repeat-busy"', '"postzero"',
        ):
            self.assertIn(fragment, source)
        shim = SHIM.read_text()
        self.assertIn("__real_et_g3c4_private_p2g2_holder_begin_capture_v1", shim)
        self.assertIn("__real_et_g3c4_private_p2g2_holder_abort_attempt_v1", shim)
        self.assertIn("ET_I64_TENSOR_CODE_ACTIVE_BORROW", shim)
        self.assertIn("ET_KERNEL_CODE_PROVIDER_REJECTED", shim)
        self.assertIn("et_g3c4_holder_tokens[et_g3c4_holder_token_count - 1].live", shim)

    def test_script_is_source_only_before_aot_and_proves_off_symbols(self):
        source = SCRIPT.read_text()
        self.assertLess(source.index("G3C4_HELD_SOURCE_ONLY"),
                        source.index("verify_toolchain"))
        self.assertIn("--emit-depfile", source)
        self.assertIn("g3c4_p2g2_terminal_held_holder_source_closure.txt", source)
        self.assertIn("feature-off object exports holder ABI", source)
        self.assertIn("-Wl,--wrap=et_g3c4_private_p2g2_holder_begin_capture_v1", source)
        self.assertIn("-Wl,--wrap=et_g3c4_private_p2g2_holder_abort_attempt_v1", source)
        self.assertIn("ESHKOL_ARENA_POISON=1", source)
        self.assertIn('test "$status" -eq 134', source)


if __name__ == "__main__":
    unittest.main()
