"""Host-only custody tests; Docker and the compiled gate are never launched."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/launch-e3-r1-stage3-compiled.py'
SPEC = importlib.util.spec_from_file_location('e3_r1_stage3_launcher', SCRIPT)
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)
CID = 'a' * 64


def container(running=False, label='owned', exit_code=0):
    return {'Id': CID, 'Config': {'Labels': {LAUNCHER.LABEL: label}},
            'State': {'Running': running, 'ExitCode': exit_code,
                      'OOMKilled': False, 'Status': 'running' if running else 'exited'}}


class Stage3LauncherCustodyTest(unittest.TestCase):
    def test_gate_uses_both_shared_locks_in_order(self):
        self.assertEqual(LAUNCHER.BUILD_LOCKS, (
            Path('/home/gabe/.cache/eshkol-transformer-orchestration-build.lock'),
            Path('/tmp/eshkol-transformer-heavy-build.lock'),
        ))
        with tempfile.TemporaryDirectory() as directory:
            paths = tuple(Path(directory) / name for name in ('outer.lock', 'legacy.lock'))
            with mock.patch.object(LAUNCHER, 'BUILD_LOCKS', paths), \
                    mock.patch.object(fcntl, 'flock', wraps=fcntl.flock) as acquire:
                with LAUNCHER.build_locks():
                    self.assertEqual([os.fstat(call.args[0]).st_ino
                                      for call in acquire.call_args_list],
                                     [path.stat().st_ino for path in paths])
                    for path in paths:
                        with path.open('r+b') as competing_lock:
                            with self.assertRaises(BlockingIOError):
                                fcntl.flock(competing_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                for path in paths:
                    with path.open('r+b') as competing_lock:
                        fcntl.flock(competing_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_gate_releases_outer_lock_when_legacy_alias_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            outer = Path(directory) / 'outer.lock'
            outer.touch()
            alias = Path(directory) / 'legacy.lock'
            os.link(outer, alias)
            with mock.patch.object(LAUNCHER, 'BUILD_LOCKS', (outer, alias)):
                with self.assertRaisesRegex(SystemExit, 'distinct regular file'):
                    with LAUNCHER.build_locks():
                        self.fail('an aliased legacy lock was acquired')
            with outer.open('r+b') as competing_lock:
                fcntl.flock(competing_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_host_qualification_preserves_all_four_floors(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory)
            (evidence / 'tmp').mkdir()
            with mock.patch.object(LAUNCHER.Path, 'read_text', return_value=
                                   'MemAvailable: 1024 kB\nSwapFree: 512 kB\n'):
                report = LAUNCHER.resource_report(evidence)
            self.assertEqual(report['host_bytes'],
                             {'MemAvailable': 1024 * 1024, 'SwapFree': 512 * 1024})
            self.assertEqual(report['container_memory_bytes'], 6 * 1024**3)
            self.assertEqual(report['container_memory_plus_swap_bytes'], 7 * 1024**3)
            self.assertEqual(report['scratch_path'], str(evidence / 'tmp'))
            self.assertEqual(report['guest_tmpdir'], '/out/tmp')
            self.assertGreater(report['scratch_free_bytes'], 0)
            report['scratch_free_bytes'] = 0
            report['host_tmp_free_bytes'] = 0
            failures = LAUNCHER.qualification_failures(report)
            self.assertEqual(failures['MemAvailable']['required_bytes'], 16 * 1024**3)
            self.assertEqual(failures['SwapFree']['required_bytes'], 8 * 1024**3)
            self.assertEqual(failures['package/output scratch']['required_bytes'], 32 * 1024**3)
            self.assertEqual(failures['/tmp']['required_bytes'], 4 * 1024**3)

    def test_qualification_accepts_exact_floor_and_rejects_one_byte_short(self):
        report = {'host_bytes': {'MemAvailable': 16 * 1024**3,
                                 'SwapFree': 8 * 1024**3},
                  'scratch_free_bytes': 32 * 1024**3,
                  'host_tmp_free_bytes': 4 * 1024**3}
        self.assertEqual(LAUNCHER.qualification_failures(report), {})
        for section, key in (('host_bytes', 'MemAvailable'),
                             ('host_bytes', 'SwapFree'),
                             (None, 'scratch_free_bytes'),
                             (None, 'host_tmp_free_bytes')):
            mutated = {'host_bytes': dict(report['host_bytes']),
                       'scratch_free_bytes': report['scratch_free_bytes'],
                       'host_tmp_free_bytes': report['host_tmp_free_bytes']}
            if section:
                mutated[section][key] -= 1
            else:
                mutated[key] -= 1
            self.assertEqual(len(LAUNCHER.qualification_failures(mutated)), 1)

    def test_dependencies_keep_their_own_git_identity(self):
        source = SCRIPT.read_text()
        self.assertNotIn('GIT_DIR=', source)
        self.assertNotIn('GIT_WORK_TREE=', source)
        self.assertIn("(SOURCE / '.deps').mkdir(exist_ok=True)", source)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project, deps = root / 'project', root / 'deps'
            for repo in (project, deps):
                subprocess.run(['git', 'init', '-q', str(repo)], check=True)
                subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Test',
                                '-c', 'user.email=test@example.invalid', 'commit',
                                '-q', '--allow-empty', '-m', repo.name], check=True)
            expected = subprocess.check_output(
                ['git', '-C', str(deps), 'rev-parse', 'HEAD'], text=True).strip()
            project_head = subprocess.check_output(
                ['git', '-C', str(project), 'rev-parse', 'HEAD'], text=True).strip()
            contaminated = {**LAUNCHER.os.environ, 'GIT_DIR': str(project / '.git'),
                            'GIT_WORK_TREE': str(project)}
            wrong = subprocess.check_output(
                ['git', '-C', str(deps), 'rev-parse', 'HEAD'], text=True,
                env=contaminated).strip()
            self.assertEqual(wrong, project_head)
            self.assertNotEqual(wrong, expected)
            actual = subprocess.check_output(
                ['git', '-C', str(deps), 'rev-parse', 'HEAD'], text=True,
                env={key: value for key, value in contaminated.items()
                     if key not in ('GIT_DIR', 'GIT_WORK_TREE')}).strip()
            self.assertEqual(actual, expected)

    def test_sanitized_fault_package_uses_fault_selector(self):
        root = SCRIPT.parents[1]
        wrapper = (root / 'scripts/build-e3-r1-stage3-test.sh').read_text()
        policy = (root / 'scripts/build-e1b-consumer.sh').read_text()
        self.assertIn('E3_FAULT_SANITIZE="$sanitize"', wrapper)
        self.assertNotIn('E3_RETENTION_SANITIZE=', wrapper)
        self.assertIn('"${e3_tuple_kind}" == fault', policy)
        self.assertIn('"${E3_FAULT_SANITIZE}" == 1', policy)

    def test_runtime_routes_stage3_partial_pin_denial(self):
        runtime = (SCRIPT.parents[1] / 'tests/e3_private/r1_stage3_runtime.esk').read_text()
        self.assertIn("(define scenario (vector 'fault 'prepare present? 4 7 1))", runtime)
        self.assertIn("(e3-fault-run scenario input output)", runtime)
        self.assertIn("E3-R1-STAGE3-PASS gradient=", runtime)

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


if __name__ == '__main__':
    unittest.main()
