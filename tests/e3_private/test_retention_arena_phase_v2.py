"""Host-only checks for the runtime-only retention phase successor."""
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
PAIR = ROOT / 'scripts/e3-retention-arena-phase-pair.sh'
GATE = ROOT / 'scripts/test-e3-retention-arena-phase-v2.sh'
RUNTIME = ROOT / 'tests/e3_private/retention_arena_diagnostic_runtime.esk'
DRIVER = ROOT / 'tests/e3_private/retention_arena_diagnostic_driver.esk'
PRIOR = Path('/home/gabe/.codex/evidence/eshkol-transformer/finish-20261022/e3/retention-arena-dd2c7ee-attempt1')
SEAL = '12d127e794607f120edc1d936e4063e1e64c50ea3dad0eded8ed42c633bd4b0e'


class RetentionArenaPhaseV2Test(unittest.TestCase):
    def test_inherited_private_driver_and_prior_receipt_are_fixed(self):
        old = subprocess.check_output(
            ['git', '-C', str(ROOT), 'show', 'dd2c7ee68c99099e5b633a27737f14639d8290ad:'
             'tests/e3_private/retention_arena_diagnostic_driver.esk'])
        self.assertEqual(DRIVER.read_bytes(), old)
        self.assertEqual(hashlib.sha256((PRIOR / 'POSTEXIT-SHA256SUMS').read_bytes()).hexdigest(), SEAL)
        guest = GATE.read_text()
        self.assertIn('prior_seal=' + SEAL, guest)
        self.assertIn('sha256sum -c POSTEXIT-SHA256SUMS', guest)
        self.assertNotIn('build-e3-retention-test.sh', guest)
        self.assertNotIn('build-e3-private.sh', guest)

    def test_samples_flush_before_inherited_report_and_teardown(self):
        old = subprocess.check_output(
            ['git', '-C', str(ROOT), 'show', 'dd2c7ee68c99099e5b633a27737f14639d8290ad:'
             'tests/e3_private/retention_arena_diagnostic_runtime.esk'], text=True)
        new = RUNTIME.read_text()
        report = '(e3-retention-test-run report input output)'
        teardown = '(e3-retention-test-run teardown input output)'
        marker = '(display "E3-RETENTION-ARENA-DIAGNOSTIC mode=")'
        flush = '(flush-output-port (current-output-port))'
        self.assertLess(new.index(marker), new.index(flush))
        self.assertLess(new.index(flush), new.index(report))
        self.assertLess(new.index(report), new.index(teardown))
        for anchor in (report, teardown, "(check 'report", "(check 'teardown"):
            self.assertIn(anchor, old)
            self.assertIn(anchor, new)
        self.assertEqual(old.count('(e3-retention-test-run diagnose input output)'), 1)
        self.assertEqual(new.count('(e3-retention-test-run diagnose input output)'), 1)

    def test_fault_nonzero_still_runs_control_in_fresh_process(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            corpus = root / 'corpus'
            corpus.mkdir()
            runtime = root / 'fake-runtime'
            runtime.write_text('''#!/usr/bin/env bash
printf '%s %s\\n' "$2" "$$" >> "$CALL_LOG"
printf 'E3-RETENTION-ARENA-DIAGNOSTIC mode=%s prepared=1 before=2 after-arm=3 after-call=4 after-checks=5 after-retry=6\\n' "$2"
if [[ "$2" == fault ]]; then
  printf 'E3-RETENTION-FAIL: balanced-i2-views-pins\\n'
  exit 23
fi
printf 'E3-RETENTION-ARENA-POSTCHECK-PASS mode=%s\\n' "$2"
''')
            runtime.chmod(0o755)
            time_tool = root / 'fake-time'
            time_tool.write_text('#!/usr/bin/env bash\nshift 4\nexec "$@"\n')
            time_tool.chmod(0o755)
            calls = root / 'calls'
            env = dict(os.environ, CALL_LOG=str(calls))
            result = subprocess.run(['bash', str(PAIR), str(root), str(runtime), str(corpus),
                                     'normal-a', '0', '10', str(time_tool)], capture_output=True, text=True, env=env)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual([line.split()[0] for line in calls.read_text().splitlines()],
                             ['fault', 'control'])
            self.assertEqual(len({line.split()[1] for line in calls.read_text().splitlines()}), 2)
            self.assertEqual(result.stdout.splitlines(),
                             ['normal-a\tfault\t23\t1\t0',
                              'normal-a\tcontrol\t0\t1\t1'])
            self.assertEqual((root / 'diagnostic-normal-a-fault.exit').read_text(), '23\n')
            self.assertIn('balanced-i2-views-pins',
                          (root / 'diagnostic-normal-a-fault.stdout').read_text())

    def test_missing_sample_remains_nonaccepting_and_control_runs(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            corpus = root / 'corpus'
            corpus.mkdir()
            runtime = root / 'fake-runtime'
            runtime.write_text('''#!/usr/bin/env bash
printf '%s\\n' "$2" >> "$CALL_LOG"
printf 'E3-RETENTION-ARENA-POSTCHECK-PASS mode=%s\\n' "$2"
''')
            runtime.chmod(0o755)
            time_tool = root / 'fake-time'
            time_tool.write_text('#!/usr/bin/env bash\nshift 4\nexec "$@"\n')
            time_tool.chmod(0o755)
            calls = root / 'calls'
            env = dict(os.environ, CALL_LOG=str(calls))
            result = subprocess.run(['bash', str(PAIR), str(root), str(runtime), str(corpus),
                                     'normal-a', '0', '10', str(time_tool)], capture_output=True, text=True, env=env)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(calls.read_text().splitlines(), ['fault', 'control'])
            self.assertEqual(result.stdout.splitlines(),
                             ['normal-a\tfault\t0\t0\t1',
                              'normal-a\tcontrol\t0\t0\t1'])

    def test_guest_and_pair_parse_and_fixed_run_matrix(self):
        subprocess.run(['bash', '-n', str(GATE)], check=True)
        subprocess.run(['bash', '-n', str(PAIR)], check=True)
        guest = GATE.read_text()
        self.assertIn('for profile in normal-a normal-b poisoned-a sanitized; do', guest)
        self.assertIn('"$profile" "$poison" "$runtime_timeout" /usr/bin/time >>"$out/STATUS.tsv" || pair_status=$?', guest)
        self.assertIn('printf \'%s\\n\' "$overall" >"$out/OVERALL.exit"', guest)


if __name__ == '__main__':
    unittest.main()
