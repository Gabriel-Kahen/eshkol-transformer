from pathlib import Path
import unittest
from tests.ci.topology import check

ROOT = Path(__file__).resolve().parents[2]
CI = ".github/workflows/ci.yml"
ACCEPTANCE = ".github/workflows/acceptance.yml"
ENGINE = ".github/workflows/full-coverage.yml"


class TopologyTests(unittest.TestCase):
    def test_current_complete_graph(self):
        self.assertEqual(check(ROOT), 18)

    def test_workflow_mutations_rejected(self):
        mutations = [
            (ENGINE, 'make "${{ matrix.test_target }}"', 'make "${{ matrix.test_target }}" || true'),
            (ENGINE, "make clean && make build", "make build"),
            (ENGINE, "timeout: 105", "timeout: 75"),
            (ENGINE, "timeout: 240", "timeout: 75"),
            (ENGINE, "A0_COMPILER_TIMEOUT_SECONDS: '60'", "A0_COMPILER_TIMEOUT_SECONDS: '600'"),
            (ENGINE, "P1_LSAN: '1'", "P1_LSAN: '0'"),
            (ENGINE, "K2_ASAN_DETECT_LEAKS: '1'", "K2_ASAN_DETECT_LEAKS: '0'"),
            (ENGINE, "if: matrix.suite == 'native-numerics'", "if: false"),
            (ENGINE, 'test "$SUITES_RESULT" = success', ': "$SUITES_RESULT"'),
            (ENGINE, "fail-fast: false", "fail-fast: true"),
            (ENGINE, "runs-on: ubuntu-22.04", "runs-on: ubuntu-latest"),
            (ENGINE, "python-version: '3.14.6'", "python-version: '3.14'"),
            (ENGINE, "ESHKOL_JOBS: '2'", "ESHKOL_JOBS: '20'"),
            (ENGINE, "actions/cache@0057852bfaa89a56745cba8c7296529d2fc39830", "actions/cache@main"),
            (ENGINE, "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1", "actions/setup-python@main"),
            (ENGINE, "test-ci-c2-load-after-build", "test-ci-c2-save-after-build"),
            (ENGINE, "scripts/ci-build-prerequisites.sh", "scripts/build.sh"),
            (CI, 'test "$TOPOLOGY_RESULT" = success', ': "$TOPOLOGY_RESULT"'),
            (CI, "needs: [topology, blocking]", "needs: topology"),
            (CI, "  workflow_dispatch:\n", ""),
            (CI, "full:success|docs:skipped|reused:skipped", "full:success|full:failure|docs:skipped|reused:skipped"),
            (CI, "evidence.py select-push", "evidence.py select-push || true"),
            (CI, "--push --base", "--base"),
            (CI, "${{ github.event.before }}", "${{ github.sha }}"),
            (CI, "  actions: read\n", ""),
            (CI, "  pull-requests: read\n", ""),
            (CI, "push:false:false:true", "push:false:false:false"),
            (CI, "${{ steps.evidence.outputs.docs_verified }}", "true"),
            (ENGINE, "test-ci-optimizer-after-build", "test-ci-core-after-build"),
            (ACCEPTANCE, 'test "$SELECTION_RESULT" = success', ': "$SELECTION_RESULT"'),
            (ACCEPTANCE, "true:skipped|false:success", "true:skipped|false:success|false:failure"),
            (ACCEPTANCE, "needs.select.outputs.reused == 'false'", "false"),
            (ACCEPTANCE, "evidence.py select", "evidence.py select || true"),
            (ACCEPTANCE, "  actions: read\n", ""),
            (ACCEPTANCE, "uses: ./.github/workflows/full-coverage.yml", "uses: ./.github/workflows/other.yml"),
        ]
        for path, before, after in mutations:
            with self.subTest(path=path, mutation=before):
                text = (ROOT / path).read_text()
                self.assertIn(before, text)
                with self.assertRaises((AssertionError, KeyError)):
                    check(ROOT, {path: text.replace(before, after, 1)})

    def test_coverage_mutations_rejected_even_if_both_tiers_drop_same_gate(self):
        text = (ROOT / "Makefile").read_text()
        for command in (
            "/usr/bin/bash scripts/test-o2.sh",
            "/usr/bin/bash scripts/test-c2.sh --group c2-operational",
            "/usr/bin/bash scripts/test-t2-boundary.sh",
            "/usr/bin/bash scripts/build-c2.sh",
            "$(MAKE) smoke-after-build benchmark-after-build",
        ):
            with self.subTest(command=command):
                self.assertIn(command, text)
                with self.assertRaises(AssertionError):
                    check(ROOT, {"Makefile": text.replace(command, ":")})


if __name__ == "__main__":
    unittest.main()
