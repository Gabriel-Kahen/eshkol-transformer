from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class F32DestroyReadinessStaticTests(unittest.TestCase):
    def test_runner_keeps_full_gate_and_feature_off_parity(self) -> None:
        runner = (ROOT / "scripts/test-g3r-f32-destroy-ready.sh").read_text()
        required = (
            "-e \"$2\" || -L \"$2\"",
            'git -C "$root" show "$baseline:native/f32_tensor.c"',
            'cmp "$out/base/f32_tensor.o" "$out/candidate/f32_tensor.o"',
            'for mode in normal repeat san; do',
            '-fsanitize=address,undefined,leak',
            'printf \'%s\\n\' "$status" >"$out/$mode/run.exit"',
            '[[ "$status" -eq 0 ]]',
            '[[ ! -s "$out/$mode/run.stderr" ]]',
            'cmp "$out/normal/run.stdout" "$out/repeat/run.stdout"',
            'cmp "$out/normal/run.stdout" "$out/san/run.stdout"',
        )
        def admissible(text: str) -> bool:
            return all(part in text for part in required)

        self.assertTrue(admissible(runner))
        for part in required:
            with self.subTest(part=part):
                self.assertFalse(admissible(runner.replace(part, "", 1)))

    def test_private_seam_is_guarded_and_registry_first(self) -> None:
        source = (ROOT / "native/f32_tensor.c").read_text()
        header = (ROOT / "native/f32_parameter_internal.h").read_text()
        self.assertIn("#ifdef ET_G3R_CANDIDATE_RETIRE_PRIVATE", source)
        self.assertIn("#ifdef ET_G3R_CANDIDATE_RETIRE_PRIVATE", header)
        seam = source.split("int32_t et_f32_tensor_private_destroy_ready_v1(", 1)[1]
        seam = seam.split("\n#endif", 1)[0]
        self.assertLess(seam.index("tensor = find_tensor(candidate);"),
                        seam.index("tensor->magic"))
        self.assertIn("f32_destroy_backing_consistent(tensor, record)", seam)
        self.assertIn("f32_destroy_backing_aliased(record)", seam)
        self.assertNotIn("et_f32_tensor_borrow_begin_v1", seam)


if __name__ == "__main__":
    unittest.main()
