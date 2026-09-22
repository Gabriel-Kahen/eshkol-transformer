from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from tests.ci.topology import check

ROOT = Path(__file__).resolve().parents[2]
CI = ".github/workflows/ci.yml"
ACCEPTANCE = ".github/workflows/acceptance.yml"
ENGINE = ".github/workflows/full-coverage.yml"


class TopologyTests(unittest.TestCase):
    def test_current_complete_graph(self):
        self.assertEqual(check(ROOT), 19)

    def test_workflow_mutations_rejected(self):
        mutations = [
            (ENGINE, 'make "${{ matrix.test_target }}"', 'make "${{ matrix.test_target }}" || true'),
            (ENGINE, "make clean && make build", "make build"),
            (ENGINE, "timeout: 105", "timeout: 75"),
            (ENGINE, "timeout: 240", "timeout: 75"),
            (ENGINE, "A0_COMPILER_TIMEOUT_SECONDS: '60'", "A0_COMPILER_TIMEOUT_SECONDS: '600'"),
            (ENGINE, "P1_LSAN: '1'", "P1_LSAN: '0'"),
            (ENGINE, "L3S_ASAN_DETECT_LEAKS: '1'", "L3S_ASAN_DETECT_LEAKS: '0'"),
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

    def test_shared_call_subgate_cannot_be_removed_or_weakened(self):
        mutations = [
            ("scripts/test-m3.sh", '/usr/bin/bash "${PROJECT_ROOT}/scripts/test-m3cg.sh"', ""),
            ("scripts/test-m3.sh", '/usr/bin/bash "${PROJECT_ROOT}/scripts/test-m3cg.sh"',
             '/usr/bin/bash "${PROJECT_ROOT}/scripts/test-m3cg.sh" || true'),
            ("scripts/test-m3cg.sh", "python3 -m unittest -v tests.m3cg.test_contract", ":"),
            ("tests/m3cg/measure_gate.py", '"test-m3cg-package.sh"', '"test-m3cg-native.sh"'),
            ("tests/m3cg/measure_gate.py", "return result.returncode", "return 0"),
            ("scripts/test-m3cg-native.sh", "detect_leaks=1", "detect_leaks=0"),
        ]
        for path, before, after in mutations:
            with self.subTest(path=path, before=before):
                text = (ROOT / path).read_text()
                self.assertIn(before, text)
                with self.assertRaises(AssertionError):
                    check(ROOT, {path: text.replace(before, after)})

    def test_g3n_inventory_and_leak_checks_are_mandatory(self):
        engine = (ROOT / ENGINE).read_text()
        make = (ROOT / "Makefile").read_text()
        entry = ("          - suite: g3n-forward\n"
                 "            test_target: test-ci-g3n-after-build\n"
                 "            timeout: 75\n")
        mutations = [
            (ENGINE, engine, entry, ""),
            (ENGINE, engine, entry, entry + entry),
            (ENGINE, engine, "G3N_ASAN_DETECT_LEAKS: '1'", "G3N_ASAN_DETECT_LEAKS: '0'"),
            ("Makefile", make, "/usr/bin/bash scripts/test-g3n.sh", ":"),
            ("Makefile", make, "/usr/bin/bash scripts/build-g3n.sh", ":"),
        ]
        for path, text, before, after in mutations:
            with self.subTest(path=path, mutation=before, replacement=after):
                self.assertIn(before, text)
                with self.assertRaises((AssertionError, KeyError)):
                    check(ROOT, {path: text.replace(before, after)})

    def test_coverage_mutations_rejected_even_if_both_tiers_drop_same_gate(self):
        text = (ROOT / "Makefile").read_text()
        for command in (
            "/usr/bin/bash scripts/test-o2.sh",
            "/usr/bin/bash scripts/test-l3s.sh",
            "/usr/bin/bash scripts/test-e3-metrics.sh",
            "/usr/bin/bash scripts/build-e3-metrics.sh",
            "/usr/bin/bash scripts/test-e3-d2.sh",
            "/usr/bin/bash scripts/test-c2.sh --group c2-operational",
            "/usr/bin/bash scripts/test-t2-boundary.sh",
            "/usr/bin/bash scripts/build-c2.sh",
            "$(MAKE) smoke-after-build benchmark-after-build",
        ):
            with self.subTest(command=command):
                self.assertIn(command, text)
                with self.assertRaises(AssertionError):
                    check(ROOT, {"Makefile": text.replace(command, ":")})

    def test_e3_d2_failure_stops_every_registered_tier(self):
        make = (ROOT / "Makefile").read_text()
        for target in ("test-after-build", "test-acceptance-predecessors-after-build",
                       "test-ci-dataset-after-build"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "Makefile").write_text(make)
                (root / "scripts").mkdir()
                for script in set(re.findall(r"scripts/([a-zA-Z0-9_.-]+\.sh)", make)):
                    (root / "scripts" / script).write_text(
                        "#!/usr/bin/bash\necho " + script + " >> called\n" +
                        ("exit 73\n" if script == "test-e3-d2.sh" else "exit 0\n"))
                result = subprocess.run(["make", target], cwd=root, capture_output=True,
                                        text=True, check=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Error 73", result.stderr)
                calls = (root / "called").read_text().splitlines()
                self.assertEqual(calls[-2:], ["test-d2.sh", "test-e3-d2.sh"])
                self.assertEqual(calls.count("test-e3-d2.sh"), 1)


if __name__ == "__main__":
    unittest.main()
