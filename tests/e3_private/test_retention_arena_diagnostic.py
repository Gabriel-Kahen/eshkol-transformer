"""Host-only custody tests; Docker and the compiled gate are never launched."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/launch-e3-retention-arena-diagnostic.py'
SPEC = importlib.util.spec_from_file_location('e3_retention_arena_diagnostic_launcher', SCRIPT)
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)
CID = 'a' * 64


def exact_diagnostic_fflush_delta(canonical: str, diagnostic: str) -> bool:
    symbols = canonical.splitlines(keepends=True)
    return ('fflush\n' not in symbols
            and symbols == sorted(set(symbols))
            and diagnostic == ''.join(sorted((*symbols, 'fflush\n'))))


def container(running=False, label='owned', exit_code=0):
    return {'Id': CID, 'Config': {'Labels': {LAUNCHER.LABEL: label}},
            'State': {'Running': running, 'ExitCode': exit_code,
                      'OOMKilled': False, 'Status': 'running' if running else 'exited'}}


class RetentionArenaDiagnosticTest(unittest.TestCase):
    def test_guest_git_environment_keeps_source_and_fe9_distinct(self):
        launcher = SCRIPT.read_text()
        self.assertNotIn("'-e', f'GIT_DIR=", launcher)
        self.assertNotIn("'-e', 'GIT_WORK_TREE=", launcher)
        self.assertIn("'-v', f'{common_git}:{common_git}:ro'", launcher)
        source = SCRIPT.parents[1]
        fe9 = LAUNCHER.DEPS / 'eshkol-src'
        if not (fe9 / '.git').is_dir():
            self.skipTest('pinned fe9 checkout is unavailable on this host')
        env = os.environ.copy()
        env.pop('GIT_DIR', None)
        env.pop('GIT_WORK_TREE', None)
        def head(path, selected_env):
            return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'],
                                           env=selected_env, text=True).strip()
        source_head = head(source, env)
        self.assertEqual(head(fe9, env),
                         'fe9dfd5241a1f4c4f58dee8442f44e4ff95e55b9')
        self.assertNotEqual(source_head, head(fe9, env))
        poisoned = dict(env, GIT_DIR=subprocess.check_output(
            ['git', '-C', str(source), 'rev-parse', '--git-dir'],
            env=env, text=True).strip(), GIT_WORK_TREE=str(source))
        self.assertEqual(head(fe9, poisoned), source_head)

    def test_separate_driver_and_exact_manifest_delta(self):
        root = SCRIPT.parents[1]
        private = root / 'tests/e3_private'
        self.assertEqual(hashlib.sha256((private / 'retention_driver.esk').read_bytes()).hexdigest(),
                         '5c2d5dcd28219e5977e574a75c983366b7fcd243e6ba100a5c9b0eee1ed9c97d')
        diagnostic = (private / 'retention_arena_diagnostic_driver.esk').read_text()
        self.assertIn('(define (e3-test-retention-diagnose', diagnostic)
        self.assertEqual((private / 'retention_arena_diagnostic_driver_root.esk').read_text(),
                         ';;; Exact test-only successor; the canonical private root remains unchanged.\n'
                         '(load "../../native/e3_private_driver_root.esk")\n'
                         '(load "retention_arena_diagnostic_driver.esk")\n')
        for original in sorted(private.glob('retention_package_*.txt')):
            candidate = private / original.name.replace('retention_package_',
                                                        'retention_arena_diagnostic_package_')
            expected = original.read_text()
            if original.name.endswith('_source_closure.txt') and not original.name.endswith('_native_source_closure.txt'):
                expected = expected.replace('retention_driver_root.esk',
                                            'retention_arena_diagnostic_driver_root.esk')
                expected = expected.replace('retention_driver.esk',
                                            'retention_arena_diagnostic_driver.esk')
            if original.name.endswith('_native_source_closure.txt'):
                expected = expected.replace('retention_bridge.c',
                                            'retention_arena_diagnostic_bridge.c')
            if original.name == 'retention_package_undefined_symbols.txt':
                self.assertTrue(exact_diagnostic_fflush_delta(expected, candidate.read_text()),
                                candidate.name)
            elif original.name == 'retention_package_sanitized_noninstrumentation_undefined_symbols.txt':
                self.assertTrue(exact_diagnostic_fflush_delta(expected, candidate.read_text()),
                                candidate.name)
            else:
                self.assertEqual(candidate.read_text(), expected, candidate.name)
        self.assertEqual((private / 'retention_arena_diagnostic_bridge.c').read_bytes(),
                         (private / 'retention_bridge.c').read_bytes())

    def test_diagnostic_normal_import_manifest_rejects_drift(self):
        private = SCRIPT.parents[1] / 'tests/e3_private'
        canonical = (private / 'retention_package_undefined_symbols.txt').read_text()
        diagnostic = (private / 'retention_arena_diagnostic_package_undefined_symbols.txt').read_text()
        self.assertTrue(exact_diagnostic_fflush_delta(canonical, diagnostic))
        for name, changed in (
            ('missing measured import', diagnostic.replace('fflush\n', '')),
            ('extra import', diagnostic + 'unmeasured_symbol\n'),
            ('same-count substitution', diagnostic.replace('fflush\n', 'unmeasured_symbol\n')),
        ):
            with self.subTest(name=name):
                self.assertFalse(exact_diagnostic_fflush_delta(canonical, changed))

    def test_diagnostic_sanitized_import_policy_rejects_drift(self):
        root = SCRIPT.parents[1]
        private = root / 'tests/e3_private'
        canonical = (private / 'retention_package_sanitized_noninstrumentation_undefined_symbols.txt').read_text()
        diagnostic = (private / 'retention_arena_diagnostic_package_sanitized_noninstrumentation_undefined_symbols.txt').read_text()
        self.assertTrue(exact_diagnostic_fflush_delta(canonical, diagnostic))
        normal = private / 'retention_arena_diagnostic_package_undefined_symbols.txt'
        instrumentation = private / 'retention_arena_diagnostic_package_sanitized_instrumentation_undefined_symbols.txt'
        policy = root / 'scripts/e3-retention-sanitizer-import-policy.sh'
        with tempfile.TemporaryDirectory() as directory:
            emitted = Path(directory) / 'noninstrumentation.txt'
            for name, content in (
                ('measured', diagnostic),
                ('missing', diagnostic.replace('fflush\n', '')),
                ('extra', diagnostic + 'unmeasured_symbol\n'),
                ('same-count', diagnostic.replace('fflush\n', 'unmeasured_symbol\n')),
            ):
                with self.subTest(name=name):
                    emitted.write_text(''.join(f'{row}\n' for row in sorted(set(content.splitlines()))))
                    result = subprocess.run(
                        ['bash', '-c', 'set -euo pipefail; PROJECT_ROOT=$1; '
                         'e3_tuple_kind=retention; e3_retention_arena_diagnostic_tuple=1; '
                         'die() { printf "%s\\n" "$*" >&2; exit 77; }; '
                         'source "$2"; e3_retention_verify_sanitizer_imports "$3" "$4" "$5"',
                         'policy', str(root), str(policy), str(normal), str(emitted),
                         str(instrumentation)], capture_output=True, text=True)
                    self.assertEqual(result.returncode == 0, name == 'measured', result.stderr)

    def test_policy_admits_only_complete_diagnostic_tuple(self):
        root = SCRIPT.parents[1]
        private = root / 'tests/e3_private'
        prefix = private / 'retention_arena_diagnostic_package'
        code = '''
PROJECT_ROOT=$1
raw_private_root=$2
raw_package_bridge=$3
raw_package_renames=$4
raw_public_exports=$5
raw_include_dirs=("$1/internal/p1/lib" "$1/internal/c1/lib" "$1/internal/t1/lib" "$1/src" "$1/internal/d2/lib" "$1/internal/e3/lib")
die() { printf '%s\\n' "$*" >&2; exit 77; }
source "$1/scripts/e3-private-package-policy.sh"
printf '%s %s %s\\n' "$e3_tuple_kind" "$e3_retention_arena_diagnostic_tuple" "$e3_prefix"
'''
        argv = ['bash', '-c', code, 'bash', str(root),
                str(private / 'retention_arena_diagnostic_driver_root.esk'),
                str(private / 'retention_arena_diagnostic_bridge.c'),
                str(prefix) + '_private_renames.txt',
                str(prefix) + '_public_exports.txt']
        accepted = subprocess.run(argv, capture_output=True, text=True)
        self.assertEqual((accepted.returncode, accepted.stdout),
                         (0, f'retention 1 {prefix}\n'), accepted.stderr)
        argv[6] = str(private / 'retention_bridge.c')
        mixed = subprocess.run(argv, capture_output=True, text=True)
        self.assertEqual(mixed.returncode, 77)
        self.assertIn('exact lexical repository tuple', mixed.stderr)

    def test_postexit_seal_covers_nested_results_and_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory)
            (evidence / 'gate').mkdir()
            (evidence / 'gate' / 'fault.stdout').write_bytes(b'fault=17\n')
            (evidence / 'result.json').write_bytes(b'{}\n')
            LAUNCHER.seal_evidence(evidence)
            lines = (evidence / 'POSTEXIT-SHA256SUMS').read_text().splitlines()
            self.assertEqual(lines, [
                hashlib.sha256(b'fault=17\n').hexdigest() + '  ./gate/fault.stdout',
                hashlib.sha256(b'{}\n').hexdigest() + '  ./result.json',
            ])
            (evidence / 'alias').symlink_to(evidence / 'result.json')
            with self.assertRaisesRegex(RuntimeError, 'unsealed symlink'):
                LAUNCHER.seal_evidence(evidence)

    def test_diagnostic_routes_fault_and_control_without_original_horizon(self):
        root = SCRIPT.parents[1]
        runtime = (root / 'tests/e3_private/retention_arena_diagnostic_runtime.esk').read_text()
        guest = (root / 'scripts/test-e3-complete-call-retention.sh').read_text()
        driver = (root / 'tests/e3_private/retention_arena_diagnostic_driver.esk').read_text()
        launcher = SCRIPT.read_text()
        self.assertIn("(vector 'retention 'diagnose 1024 #f fault?)", runtime)
        self.assertIn('for mode in fault control; do', guest)
        self.assertIn("'scripts/test-e3-complete-call-retention.sh'", launcher)
        self.assertIn("'/out/gate', 'diagnostic', args.expected_head, args.expected_tree", launcher)
        self.assertNotIn("(evidence / 'gate').mkdir()", launcher)
        self.assertIn('[[ ! -e "${out}" ]]', guest)
        self.assertLess(guest.index('if [[ "$diagnostic" == 1 ]]; then'),
                        guest.index('for horizon in 1024 8192; do'))
        self.assertIn('(vector-set! samples 0 (__arena-used))', driver)
        for index in range(1, 5):
            self.assertIn(f'(vector-set! (vector-ref record 19) {index} (__arena-used))', driver)

    def test_fault_probes_bracket_failure_checks_and_retry(self):
        driver = (SCRIPT.parents[1] / 'tests/e3_private/retention_arena_diagnostic_driver.esk').read_text()
        failure = driver.split('(define (e3-test-retention-fail-retry!', 1)[1]
        failure = failure.split('(define (e3-test-retention-reconcile!', 1)[0]
        anchors = [
            '(e3-test-read-fail-stage 1)',
            '(vector-set! (vector-ref record 19) 1 (__arena-used))',
            "(with-region ('e3-retention-failure 262144)",
            '(vector-set! (vector-ref record 19) 2 (__arena-used))',
            '(e3-test-read-fail-stage 0)',
            "(with-region ('e3-retention-failure-check 262144)",
            '(vector-set! (vector-ref record 19) 3 (__arena-used))',
            '(e3-test-retention-call! model dataset frame destinations cursor modes',
            '(vector-set! (vector-ref record 19) 4 (__arena-used))',
        ]
        positions = [failure.index(anchor) for anchor in anchors]
        self.assertEqual(positions, sorted(positions))
        diagnostic = driver.split('(define (e3-test-retention-diagnose', 1)[1]
        diagnostic = diagnostic.split('(define (e3-test-retention-report', 1)[0]
        self.assertIn('(if fault?\n        (e3-test-retention-fail-retry!', diagnostic)
        self.assertIn("(with-region ('e3-retention-control-check 262144)", diagnostic)

    def test_stopped_owned_cid_is_removed_before_absence_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory)
            (evidence / 'container.cid').write_text(CID + '\n')
            with mock.patch.object(LAUNCHER, 'inspect_container', side_effect=[
                    container(exit_code=17), None, None]) as inspect, \
                    mock.patch.object(LAUNCHER.subprocess, 'run') as command, \
                    mock.patch.object(LAUNCHER.time, 'sleep'):
                custody = LAUNCHER.ensure_container_gone(evidence, 'owned-name', 'owned')
            self.assertEqual([call.args[0] for call in inspect.call_args_list],
                             [CID, CID, 'owned-name'])
            self.assertEqual(command.call_args.args[0], ['docker', 'rm', '--force', CID])
            self.assertEqual(custody['exit_state']['exit_code'], 17)
            self.assertTrue(custody['owned_absent'])
            self.assertFalse(custody['force_stopped'])
            self.assertEqual(json.loads((evidence / 'container-custody.json').read_text()), custody)

    def test_running_owned_cid_is_stopped_and_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory)
            (evidence / 'container.cid').write_text(CID)
            with mock.patch.object(LAUNCHER, 'inspect_container', side_effect=[
                    container(running=True), None, None]), \
                    mock.patch.object(LAUNCHER.subprocess, 'run') as command, \
                    mock.patch.object(LAUNCHER.time, 'sleep'):
                custody = LAUNCHER.ensure_container_gone(evidence, 'owned-name', 'owned')
            self.assertEqual([call.args[0][:2] for call in command.call_args_list],
                             [['docker', 'stop'], ['docker', 'rm']])
            self.assertTrue(custody['force_stopped'])
            self.assertIsNone(custody['exit_state'])

    def test_foreign_name_is_never_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory)
            with mock.patch.object(LAUNCHER, 'inspect_container',
                                   return_value=container(label='foreign')) as inspect, \
                    mock.patch.object(LAUNCHER.subprocess, 'run') as command:
                custody = LAUNCHER.ensure_container_gone(evidence, 'candidate-name', 'owned')
            inspect.assert_called_once_with('candidate-name')
            command.assert_not_called()
            self.assertTrue(custody['foreign_name'])

    def test_cid_label_mismatch_retries_without_removing_foreign(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory)
            (evidence / 'container.cid').write_text(CID)
            with mock.patch.object(LAUNCHER, 'inspect_container', side_effect=[
                    container(label='foreign'), container(label='owned'), None, None]), \
                    mock.patch.object(LAUNCHER.subprocess, 'run') as command, \
                    mock.patch.object(LAUNCHER.time, 'sleep'):
                custody = LAUNCHER.ensure_container_gone(evidence, 'owned-name', 'owned')
            self.assertEqual(command.call_count, 1)
            self.assertEqual(command.call_args.args[0], ['docker', 'rm', '--force', CID])
            self.assertIn('CID ownership mismatch',
                          (evidence / 'container-custody-retry.log').read_text())
            self.assertTrue(custody['owned_absent'])

    def test_stale_cid_falls_back_to_labeled_name_then_removes_it(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory)
            (evidence / 'container.cid').write_text(CID)
            with mock.patch.object(LAUNCHER, 'inspect_container', side_effect=[
                    None, container(label='owned'), None, None]), \
                    mock.patch.object(LAUNCHER.subprocess, 'run') as command, \
                    mock.patch.object(LAUNCHER.time, 'sleep'):
                custody = LAUNCHER.ensure_container_gone(evidence, 'owned-name', 'owned')
            self.assertEqual(command.call_count, 1)
            self.assertTrue(custody['owned_absent'])
            self.assertTrue(custody['identity_mismatch'])

    def test_malformed_cid_checks_name_and_cannot_pass_custody(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory)
            (evidence / 'container.cid').write_text('bad-cid')
            with mock.patch.object(LAUNCHER, 'inspect_container', return_value=None) as inspect, \
                    mock.patch.object(LAUNCHER.time, 'sleep'):
                custody = LAUNCHER.ensure_container_gone(evidence, 'owned-name', 'owned')
            self.assertEqual(inspect.call_count, 3)
            self.assertTrue(custody['identity_mismatch'])
            self.assertEqual(custody['cid'], 'bad-cid')

    def test_missing_cid_requires_three_name_absence_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory)
            with mock.patch.object(LAUNCHER, 'inspect_container', return_value=None) as inspect, \
                    mock.patch.object(LAUNCHER.time, 'sleep') as pause:
                custody = LAUNCHER.ensure_container_gone(evidence, 'owned-name', 'owned')
            self.assertEqual(inspect.call_count, 3)
            self.assertEqual(pause.call_count, 2)
            self.assertEqual(custody['nameless_absent_checks'], 3)

    def test_cancelled_start_process_group_is_reaped(self):
        child = mock.Mock(pid=1234)
        child.poll.return_value = None
        child.wait.side_effect = [subprocess.TimeoutExpired('docker start', 60), 0]
        with mock.patch.object(LAUNCHER.os, 'killpg') as kill:
            LAUNCHER.stop_child(child)
        self.assertEqual(kill.call_args_list,
                         [mock.call(1234, signal.SIGTERM), mock.call(1234, signal.SIGKILL)])
        self.assertEqual(child.wait.call_count, 2)

    def test_dependency_mountpoint_is_empty_real_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(LAUNCHER, 'SOURCE', root):
                self.assertEqual(LAUNCHER.prepare_dependency_mountpoint(), root / '.deps')
                self.assertTrue((root / '.deps').is_dir())
                (root / '.deps' / 'foreign').write_text('preserve')
                with self.assertRaisesRegex(RuntimeError, 'empty real directory'):
                    LAUNCHER.prepare_dependency_mountpoint()
                (root / '.deps' / 'foreign').unlink()
                (root / '.deps').rmdir()
                (root / '.deps').symlink_to(root)
                with self.assertRaisesRegex(RuntimeError, 'empty real directory'):
                    LAUNCHER.prepare_dependency_mountpoint()


if __name__ == '__main__':
    unittest.main()
