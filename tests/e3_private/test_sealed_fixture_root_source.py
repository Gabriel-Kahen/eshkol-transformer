#!/usr/bin/env python3
"""Static drift checks for the unrun, noninstalled sealed-M3 root closure."""

from pathlib import Path
import re
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


def source(path: str) -> str:
    return (ROOT / path).read_text()


def loads(text: str) -> list[str]:
    return re.findall(r'^\(load "([^"]+)"\)$', text, re.MULTILINE)


class SealedFixtureRootSource(unittest.TestCase):
    def test_fixture_flattens_canonical_root_order_once(self) -> None:
        fixture = source("tests/e3_private/sealed_fixture_driver_root.esk")
        canonical = (
            loads(source("native/m3t_package_root.esk"))
            + loads(source("native/m3_package_root.esk"))[1:]
            + loads(source("native/e3_private_root.esk"))[1:]
            + loads(source("native/e3_private_driver_root.esk"))[1:]
            + ["sealed_fixture_driver.esk"]
        )
        expected = [name if name in ("i2_wave2_root.esk",
                                      "m3t_transport_extension.esk")
                    else "driver.esk" if name == "../tests/e3_private/driver.esk"
                    else "../../native/" + name if (ROOT / "native" / name).is_file()
                    else name for name in canonical]
        self.assertEqual(loads(fixture), expected)
        self.assertEqual(loads(source("native/i2_wave2_root.esk"))[0],
                         "t1_wave1_root.esk")
        self.assertIn("(require transformer.module)",
                      source("native/t1_wave1_root.esk"))
        self.assertNotIn("e3_private_driver_root.esk", fixture)
        with self.assertRaises(AssertionError):
            self.assertEqual(loads(fixture.replace('(load "m3t_transport_extension.esk")',
                                                  '(load "../../native/m3t_transport_extension.esk")')),
                             expected)

    def test_generated_i2_and_m3t_resolve_from_fixture_directory(self) -> None:
        fixture_dir = ROOT / "tests/e3_private"
        native_dir = ROOT / "native"
        with tempfile.TemporaryDirectory() as temp:
            overlay = Path(temp) / "source"
            overlay.mkdir()
            compiler = Path(temp) / "compiler"
            compiler.mkdir()
            for name in ("i2_wave2_root.esk", "m3t_transport_extension.esk"):
                self.assertFalse((fixture_dir / name).exists())
                self.assertFalse((ROOT / name).exists())
                self.assertTrue((native_dir / name).exists())
                (overlay / name).write_text("generated")

            def resolved(base: Path, name: str) -> Path:
                return next(path for path in (base / name, compiler / name, overlay / name,
                                              native_dir / name) if path.exists())

            fixture_loads = loads(source("tests/e3_private/sealed_fixture_driver_root.esk"))
            native_loads = loads(source("native/m3t_package_root.esk"))
            for index, name in enumerate(("i2_wave2_root.esk",
                                          "m3t_transport_extension.esk")):
                self.assertEqual(resolved(fixture_dir, fixture_loads[index]), overlay / name)
                self.assertEqual(resolved(native_dir, native_loads[index]), native_dir / name)

    def test_capture_and_package_compile_the_same_fixture_root(self) -> None:
        root = '"${PROJECT_ROOT}/tests/e3_private/sealed_fixture_driver_root.esk"'
        self.assertIn(root, source("scripts/capture-e3-sealed-fixture-source.sh"))
        self.assertIn(root, source("scripts/build-e3-sealed-fixture-test.sh"))

    def test_exact_composed_source_closure(self) -> None:
        zero = source("tests/e3_private/zero_control_package_source_closure.txt").splitlines()
        replacements = {
            "tests/e3_private/zero_control_driver_root.esk":
                "tests/e3_private/sealed_fixture_driver_root.esk",
            "native/i2_wave2_root.esk": "@BUILD@/source/i2_wave2_root.esk",
            "internal/p1/lib/transformer/module.esk":
                "@BUILD@/source/transformer/module.esk",
            "tests/e3_private/zero_control_driver.esk":
                "tests/e3_private/sealed_fixture_driver.esk",
        }
        self.assertEqual(len(zero), 33)
        for original in replacements:
            self.assertEqual(zero.count(original), 1)
        self.assertEqual(zero.count("tests/e3_private/zero_control_negative_driver.esk"), 1)
        bypassed_roots = {"native/e3_private_driver_root.esk",
                          "native/e3_private_root.esk", "native/m3_package_root.esk",
                          "native/m3t_package_root.esk",
                          "tests/e3_private/zero_control_negative_driver.esk"}
        expected = [replacements.get(row, row) for row in zero
                    if row not in bypassed_roots]
        self.assertEqual(len(expected), 28)
        actual = source("tests/e3_private/sealed_fixture_package_source_closure.txt")
        self.assertEqual(actual, "\n".join(expected) + "\n")
        self.assertEqual(len(actual.splitlines()), len(set(expected)))
        for row in ("@BUILD@/source/i2_wave2_root.esk",
                    "@BUILD@/source/transformer/module.esk",
                    "@BUILD@/source/m3t_transport_extension.esk",
                    "@BUILD@/source/e3_d2_dataset.esk"):
            self.assertEqual(expected.count(row), 1)

    def test_private_stage_precedes_compilation_and_normalizes_all_four(self) -> None:
        builder = source("scripts/build-e1b-consumer.sh")
        policy = source("scripts/e3-private-package-policy.sh")
        stage = builder.index('if [[ "${e3_tuple_kind}" == sealed-fixture ]]; then\n'
                              '    fixture_overlay=')
        compile_private = builder.index('  run_compiler --strict-types --no-stdlib')
        self.assertLess(stage, compile_private)
        self.assertIn('include_args=(-I "${e3_generated_source_root}" '
                      '"${include_args[@]}")', builder)
        stage_source = builder[stage:compile_private]
        self.assertIn('for fixture_source in transformer/module.esk i2_wave2_root.esk',
                      stage_source)
        self.assertIn('m3t_transport_extension.esk; do', stage_source)
        self.assertIn('cp -- "${fixture_overlay}/${fixture_source}"', stage_source)
        self.assertIn('cmp -- "${fixture_overlay}/${fixture_source}"', stage_source)
        self.assertIn('"${e3_generated_source_root}/${fixture_source}"', stage_source)
        for row in ("transformer/module.esk", "i2_wave2_root.esk",
                    "m3t_transport_extension.esk"):
            self.assertIn(f"@BUILD@/source/{row}", policy)
        self.assertIn("@BUILD@/source/e3_d2_dataset.esk", policy)
        self.assertIn('cmp -s "${package_source_closure}" '
                      '"${e1b_tmp}/source-closure.txt"', builder)


if __name__ == "__main__":
    unittest.main()
