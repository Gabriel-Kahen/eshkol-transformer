"""Guard the native C2/O2 fixture inclusion in the blocking CI suite."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/test-c2-o2-native-fixtures.sh"
FIXTURES = (
    "tests/g3r/same_owner_o2_hold_test.c",
    "tests/c2/save_image_receipt_native.c",
)


def missing_fixture_sources(runner: str) -> set[str]:
    return {source for source in FIXTURES if f'"$root/{source}"' not in runner}


class NativeFixtureWiringTests(unittest.TestCase):
    def test_both_fixtures_reach_blocking_ci(self) -> None:
        runner = RUNNER.read_text()
        makefile = (ROOT / "Makefile").read_text()
        workflow = (ROOT / ".github/workflows/full-coverage.yml").read_text()
        self.assertFalse(missing_fixture_sources(runner))
        target = makefile.split("test-ci-c2-save-after-build:\n", 1)[1].split("\n\n", 1)[0]
        self.assertIn("scripts/test-c2-o2-native-fixtures.sh", target)
        self.assertIn("test_target: test-ci-c2-save-after-build", workflow)
        self.assertIn("for mode in normal repeat san; do", runner)

    def test_omitting_either_fixture_fails_the_wiring_check(self) -> None:
        runner = RUNNER.read_text()
        self.assertFalse(missing_fixture_sources(runner))
        for source in FIXTURES:
            with self.subTest(source=source):
                omitted = runner.replace(f'"$root/{source}"', '"$root/omitted.c"')
                self.assertIn(source, missing_fixture_sources(omitted))


if __name__ == "__main__":
    unittest.main()
