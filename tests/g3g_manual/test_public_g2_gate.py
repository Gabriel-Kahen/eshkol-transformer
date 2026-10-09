"""Owned controls for installed package and sanitizer source selection."""

from pathlib import Path
import unittest

from tests.g3g_manual.check_public_g2_gate import check


ROOT = Path(__file__).resolve().parents[2]
BUILDER = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
RUNNER = (ROOT / "scripts/test-g3g-manual-public-g2.sh").read_text()
CALLER = (ROOT / "tests/g3g_manual/public_g2_runtime.esk").read_text()


class GateSource(unittest.TestCase):
    def test_selected_source(self):
        check(BUILDER, RUNNER, CALLER)

    def test_wrong_or_missing_installed_sanitizer_is_denied(self):
        changes = (
            (BUILDER.replace('"${package_policy}" == g3g-manual-public-aggregate &&\n'
                             '       "${e3_private_test_sanitize}" == 0',
                             '"${package_policy}" == e3-private-aggregate &&\n'
                             '       "${e3_private_test_sanitize}" == 0', 1), RUNNER, CALLER),
            (BUILDER.replace('package_native_cflags+=("${package_test_sanitize_flags[@]}")',
                             ':', 1), RUNNER, CALLER),
            (BUILDER.replace('"${package_test_sanitize_flags[@]}" \\\n  "${package_bridge_flags[@]}"',
                             '"${package_bridge_flags[@]}"', 1), RUNNER, CALLER),
            (BUILDER, RUNNER.replace('G3G_MANUAL_SANITIZE=1 /usr/bin/bash',
                                     'G3G_MANUAL_SANITIZE=0 /usr/bin/bash', 1), CALLER),
            (BUILDER, RUNNER.replace('ASAN_OPTIONS=detect_leaks=1:halt_on_error=1',
                                     'ASAN_OPTIONS=detect_leaks=0:halt_on_error=1', 1), CALLER),
            (BUILDER, RUNNER, CALLER + '\n(load "g3g_manual_package_root.esk")\n'),
            (BUILDER, RUNNER.replace('G2 evidence directory already exists',
                                     'evidence already exists', 1), CALLER),
        )
        for builder, runner, caller in changes:
            with self.subTest(mutant=(builder != BUILDER, runner != RUNNER,
                                      caller != CALLER)), self.assertRaises(ValueError):
                check(builder, runner, caller)


if __name__ == "__main__":
    unittest.main()
