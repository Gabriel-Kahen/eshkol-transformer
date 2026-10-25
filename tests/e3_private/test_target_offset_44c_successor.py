"""Pin the exact 44c native inputs consumed by the N3 target-offset successor."""

from hashlib import sha256
from pathlib import Path
import unittest

from tests.e3_reference.target_offset import materialize


ROOT = Path(__file__).resolve().parents[2]
NATIVE = {
    "src/eshkol_transformer/e3_frame.c": "848317acc2d5b08c39143cfa0af0e41acdf9f8ccb3c4b72e10e515d0d2d66864",
    "src/eshkol_transformer/e3_frame_internal.h": "678186cf1926d5e39e66aa298b7db294f9b29f780a08b14ce4aca7612e13de0b",
    "src/eshkol_transformer/m3_call_f32_integration.c": "088ec6fda152740910c0a5eb491bcb1db77396cb12a9f45f843af9ec144a32e5",
}


class TargetOffset44cSuccessor(unittest.TestCase):
    def test_native_inputs_match_44c(self) -> None:
        for name, expected in NATIVE.items():
            with self.subTest(name=name):
                self.assertEqual(sha256((ROOT / name).read_bytes()).hexdigest(), expected)

    def test_genuine_pair_changes_only_second_target(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="e3-n3-44c-") as temporary:
            cases = materialize(Path(temporary) / "corpus")["cases"]
        first, second = (case["batches"][0] for case in cases)
        self.assertEqual(first["inputs"], second["inputs"])
        self.assertEqual(first["mask"], second["mask"])
        self.assertEqual(first["targets"], [197, 0])
        self.assertEqual(second["targets"], [197, 7])


if __name__ == "__main__":
    unittest.main()
