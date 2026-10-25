"""Source gate for the fixture's single f32 testing owner."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/test-g3c4-p2g2-typed-aggregate-smoke.sh"
F32 = ROOT / "native/f32_tensor.c"


def has_test_owner(runner: str) -> bool:
    return all((
        "native/f32_tensor.c" in runner,
        "f32_tensor) extra=(-DET_F32_TENSOR_TESTING)" in runner,
        "libg3c4_p2g2_typed_aggregate.a(f32_tensor.o)" in runner,
    ))


class TypedAggregatePackageSource(unittest.TestCase):
    def test_f32_testing_owner_closes_transport_test_hooks(self):
        runner = RUNNER.read_text()
        source = F32.read_text()
        self.assertTrue(has_test_owner(runner))
        for name in (
            "et_f32_tensor_test_metadata_snapshot_v1",
            "et_f32_tensor_test_metadata_corrupt_v1",
            "et_f32_tensor_test_metadata_restore_v1",
            "et_f32_tensor_test_fail_alloc_after_v1",
            "et_f32_test_live_counts_snapshot_v1",
        ):
            self.assertIn(name, source)
        self.assertFalse(has_test_owner(
            runner.replace("native/f32_tensor.c", "native/data_io.c", 1)))
        self.assertFalse(has_test_owner(
            runner.replace("f32_tensor) extra=(-DET_F32_TENSOR_TESTING)",
                           "f32_tensor) extra=()", 1)))


if __name__ == "__main__":
    unittest.main()
