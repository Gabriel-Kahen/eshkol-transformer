#!/usr/bin/env python3
"""Static drift checks for the unrun, noninstalled sealed-M3 root closure."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


def source(path: str) -> str:
    return (ROOT / path).read_text()


class SealedFixtureRootSource(unittest.TestCase):
    def test_one_root_chain_and_fixture_entry(self) -> None:
        loads = (
            ("tests/e3_private/sealed_fixture_driver_root.esk",
             '(load "../../native/e3_private_driver_root.esk")',
             '(load "sealed_fixture_driver.esk")'),
            ("native/e3_private_driver_root.esk", '(load "e3_private_root.esk")'),
            ("native/e3_private_root.esk", '(load "m3_package_root.esk")'),
            ("native/m3_package_root.esk", '(load "m3t_package_root.esk")'),
            ("native/m3t_package_root.esk", '(load "i2_wave2_root.esk")',
             '(load "m3t_transport_extension.esk")'),
            ("native/i2_wave2_root.esk", '(load "t1_wave1_root.esk")'),
            ("native/t1_wave1_root.esk", '(require transformer.module)'),
        )
        for path, *forms in loads:
            text = source(path)
            for form in forms:
                with self.subTest(path=path, form=form):
                    self.assertEqual(text.count(form), 1)
        fixture_root = source(loads[0][0])
        self.assertLess(fixture_root.index(loads[0][1]),
                        fixture_root.index(loads[0][2]))

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
        expected = [replacements.get(row, row) for row in zero
                    if row != "tests/e3_private/zero_control_negative_driver.esk"]
        self.assertEqual(len(expected), 32)
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
