"""Negative source/ABI checks for the current-main private capture port."""
from pathlib import Path
from hashlib import sha256
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "native/g3r_same_owner_save.h"
COORDINATOR = ROOT / "native/g3r_same_owner_save.c"
CALLER = ROOT / "native/g3r_same_owner_save_caller_extension.esk"
ADOPTION = ROOT / "native/g3r_same_owner_o2_adoption_extension.esk"


def compile_header(extra_flags):
    source = r'''
#include <stddef.h>
#include "native/g3r_same_owner_save.h"
_Static_assert(sizeof(et_g3r_committed_history_snapshot_v1) == 136, "image");
_Static_assert(offsetof(et_g3r_committed_history_snapshot_v1, policy) == 0, "policy");
_Static_assert(offsetof(et_g3r_committed_history_snapshot_v1, rng) == 48, "rng");
_Static_assert(offsetof(et_g3r_committed_history_snapshot_v1, ids) == 80, "ids");
_Static_assert(offsetof(et_g3r_committed_history_snapshot_v1, length) == 96, "length");
_Static_assert(offsetof(et_g3r_committed_history_snapshot_v1, prompt_length) == 104, "prompt");
_Static_assert(offsetof(et_g3r_committed_history_snapshot_v1, generated_count) == 112, "generated");
_Static_assert(offsetof(et_g3r_committed_history_snapshot_v1, manual_count) == 120, "manual");
_Static_assert(offsetof(et_g3r_committed_history_snapshot_v1, generated_eos_terminal) == 128, "terminal");
_Static_assert(sizeof(et_g3r_g3t_held_words_v1) == 136, "held image");
'''
    return subprocess.run(
        ["cc", "-std=c11", "-fsyntax-only", "-x", "c", "-", "-I.", "-Isrc",
         "-DET_G3R_SAME_OWNER_SAVE_PRIVATE=1", *extra_flags],
        input=source, text=True, capture_output=True, cwd=ROOT)


class CurrentMainCapturePort(unittest.TestCase):
    def test_current_history_shape_and_offsets_compile(self):
        result = compile_header(["-DET_G3R_COMMITTED_HISTORY_PRIVATE=1"])
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_old_ledger_macro_cannot_enable_capture(self):
        result = compile_header(["-DET_G3R_G3T_SAVE_LEDGER_PRIVATE=1",
                                 "-DET_G3R_COMMITTED_HISTORY_PRIVATE=1"])
        # The obsolete macro must be rejected explicitly, even if the new
        # committed-history feature is also set.
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("obsolete", result.stderr)

    def test_recursive_closure_matches_current_main_manifest(self):
        closure = (ROOT / "native/g3r_same_owner_o2_adoption_source_closure.txt")
        paths = closure.read_text().splitlines()
        self.assertEqual(len(paths), 96)
        self.assertEqual(len(set(paths)), len(paths))
        manifest = (ROOT / "native/g3r_same_owner_capture_main_source.sha256")
        rows = [row.split("  ", 1) for row in manifest.read_text().splitlines()]
        self.assertEqual([path for _, path in rows], paths)
        for digest, path in rows:
            with self.subTest(path=path):
                self.assertEqual(sha256((ROOT / path).read_bytes()).hexdigest(),
                                 digest)

    def test_capture_reads_only_seventeen_owned_words(self):
        coordinator = COORDINATOR.read_text()
        adoption = ADOPTION.read_text()
        caller = CALLER.read_text()
        self.assertIn("17u * sizeof(int64_t)", coordinator)
        self.assertIn("index >= 17", coordinator)
        self.assertIn("(make-vector 17 0)", adoption)
        self.assertIn("(< index 17)", adoption)
        self.assertIn("(vector-length words) 17", caller)
        self.assertIn("(vector-ref words 12)", caller)
        self.assertNotIn("idle_snapshot", coordinator)
        self.assertNotIn("G3R_G3T_SAVE_LEDGER_PRIVATE", coordinator)


if __name__ == "__main__":
    unittest.main()
