"""Source contract gate for the private P2/G2 Eshkol terminal workstream.

This checks the exact upstream ABI and the predecessor boundary. It is not a
same-call execution test and cannot establish terminal publication.
"""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "src/eshkol_transformer/g3c4_context_internal.h"
NATIVE = ROOT / "src/eshkol_transformer/g3c4_model_owner.c"
SCOPE = ROOT / "native/g3c4_p2g2_protected_scope_extension.esk"
PREFIX = ROOT / "native/g3c4_p2g2_t1_prefix_coordinator_extension.esk"


class TerminalSourcePrerequisites(unittest.TestCase):
    def test_exact_private_native_abi_is_declared_and_defined(self):
        declarations = HEADER.read_text()
        definitions = NATIVE.read_text()
        expected = {
            "et_g3c4_private_p2g2_eos_first_frame_carrier_v1":
                "void *context, void *input, void *staging_header, int64_t carrier_bytes",
            "et_g3c4_private_p2g2_second_frame_carrier_v1":
                "void *context, void *pending_output, void *staging_header, int64_t carrier_bytes",
            "et_g3c4_private_p2g2_terminal_commit_v1":
                "void *context, void *pending_output, const void *first_raw_header, int64_t first_raw_bytes, const void *second_raw_header, int64_t second_raw_bytes",
            "et_g3c4_private_p2g2_terminal_snapshot_v1":
                "void *published_output, void *destination_header, int64_t destination_bytes",
            "et_g3c4_private_p2g2_terminal_ids_clone_v1":
                "void *terminal_output",
            "et_g3c4_private_p2g2_terminal_ids_clone_copy_v1":
                "void *clone, int64_t expected_length, void *destination_header, int64_t destination_bytes",
            "et_g3c4_private_p2g2_terminal_ids_clone_release_v1":
                "void *clone",
        }
        for name, parameters in expected.items():
            result = "void *" if name.endswith("ids_clone_v1") else "int64_t"
            normalized = r"\s*".join(map(re.escape, parameters.split()))
            signature = rf"{re.escape(result)}\s*{name}\s*\(\s*{normalized}\s*\)"
            with self.subTest(name=name):
                self.assertRegex(declarations, signature)
                # C definitions may use more descriptive parameter names;
                # the declaration fixes the ABI while this proves a body exists.
                self.assertRegex(definitions, rf"{re.escape(result)}\s*{name}\s*\(")
        self.assertIn("#ifdef ET_G3C4_P2_G2_TERMINAL_PRIVATE", declarations)
        self.assertIn(
            "#ifdef ET_G3C4_P2_G2_TERMINAL_IDS_CLONE_PRIVATE", declarations
        )

    def test_existing_eshkol_scope_is_abort_required(self):
        scope = SCOPE.read_text()
        prefix = PREFIX.read_text()
        self.assertIn('"P2/G2 scope must abort in its call region"', scope)
        self.assertIn("(g3c4-p2g2-protected-abort! call)", scope)
        self.assertIn("(eq? (vector-ref record 13) 'before-prefix)", prefix)
        self.assertNotIn("et_g3c4_private_p2g2_terminal_commit_v1", scope)
        self.assertNotIn("et_g3c4_private_p2g2_terminal_commit_v1", prefix)


if __name__ == "__main__":
    unittest.main()
