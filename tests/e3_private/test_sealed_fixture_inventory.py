"""The extraction tool separates emitted evidence from the static candidate."""

from pathlib import Path
import importlib.util
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "sealed_inventory", ROOT / "scripts/extract-e3-sealed-fixture-inventory.py")
assert SPEC and SPEC.loader
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


class Inventory(unittest.TestCase):
    def test_raw_depfile_order_and_generated_overlay(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            gate = Path(temp)
            generated = gate / "source"
            depfile = gate / "private.d"
            expected = (ROOT / "tests/e3_private/sealed_fixture_driver_root.esk",
                        generated / "i2_wave2_root.esk",
                        generated / "transformer/module.esk")
            depfile.write_text("private: " + " \\\n  ".join(map(str, expected)) + "\n")
            self.assertEqual(TOOL.source_closure(ROOT, generated, depfile),
                             ("tests/e3_private/sealed_fixture_driver_root.esk\n"
                              "@BUILD@/source/i2_wave2_root.esk\n"
                              "@BUILD@/source/transformer/module.esk\n").encode())
            depfile.write_text(f"private: {expected[0]} {expected[0]}\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                TOOL.source_closure(ROOT, generated, depfile)

    def test_foreign_input_and_missing_object_evidence_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            gate = Path(temp)
            depfile = gate / "private.d"
            depfile.write_text("private: /tmp/foreign.esk\n")
            with self.assertRaisesRegex(ValueError, "foreign"):
                TOOL.source_closure(ROOT, gate / "source", depfile)
            evidence = gate / "object.evidence"
            evidence.mkdir()
            output = gate / "observed"
            with self.assertRaisesRegex(ValueError, "incomplete"):
                TOOL.object_inventory(evidence, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
