"""Host-only custody tests; Docker and the compiled gate are never launched."""
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/launch-e3-r1-stage8-compiled.py'
SPEC = importlib.util.spec_from_file_location('e3_r1_stage8_launcher', SCRIPT)
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)
CID = 'a' * 64


def container(running=False, label='owned', exit_code=0):
    return {'Id': CID, 'Config': {'Labels': {LAUNCHER.LABEL: label}},
            'State': {'Running': running, 'ExitCode': exit_code,
                      'OOMKilled': False, 'Status': 'running' if running else 'exited'}}


class Stage8LauncherCustodyTest(unittest.TestCase):
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
