"""Deterministic custody/control tests; fake Docker never runs an Eshkol build."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import signal
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tests.g3r import one_use_gate as gate
from tests.g3r.prepare_pair_file_inputs import record as companion_record
from tests.c2.test_checkpoint_format import make_fixture


def write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def binary(sanitized=False):
    data = bytearray(64)
    data[:6] = b'\x7fELF\x02\x01'
    struct.pack_into('<HH', data, 16, 2, 62)
    return bytes(data) + (b'__asan_init\0__ubsan_handle_add_overflow\0' if sanitized else b'')


def trace():
    return (b'G3R_ORACLE_PRE categorical 1729 0 0 ' + b' '.join([b'0'] * 256)
            + b'\nG3R_ORACLE_POST 217 1 1 1729 1729 1 1 0 0\n'
            + b'G3-R replayed P1/G1 private PASS: checks=9\n')


class Fixture:
    def __init__(self, root: Path, profile='replay'):
        self.root, self.profile = root, profile
        self.package = root / 'package'
        self.prep = self.package / 'preparation'
        self.prep.mkdir(parents=True)
        self.source = root / 'source'
        self.source.mkdir()
        for args in (['init', '-q'], ['config', 'user.name', 'Fixture'],
                     ['config', 'user.email', 'fixture@example.invalid']):
            subprocess.run(['git', '-C', str(self.source), *args], check=True)
        write(self.source / 'fixture.esk', b'(display "generated fixture")\n')
        subprocess.run(['git', '-C', str(self.source), 'add', '.'], check=True)
        subprocess.run(['git', '-C', str(self.source), 'commit', '-qm', 'fixture'], check=True)
        self.client = root / 'docker-fixture'
        write(self.client, b'fake Docker, never executed\n')
        self.sockpath = root / 'docker.sock'
        self.sock = socket.socket(socket.AF_UNIX)
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            # Kernel resolves the directory descriptor; fixture TMPDIR paths
            # under the worktree exceed AF_UNIX's108-byte address limit.
            self.sock.bind(f'/proc/self/fd/{fd}/docker.sock')
        finally:
            os.close(fd)
        self.docker_root = root / 'docker-root'
        self.docker_root.mkdir()
        deps = []
        for name in ('eshkol-src', 'eshkol-build'):
            path = root / name
            path.mkdir()
            write(path / 'pin', name.encode())
            deps.append({'root': str(path), 'guest': '/workspace/.deps/' + name,
                         'files': gate.manifest(path)})
        steps = ['pins-before', 'toolchain', 'pins-after']
        if profile == 'replay':
            steps += [f'{mode}-{action}' for mode in ('normal', 'repeat', 'sanitize') for action in ('run', 'oracle')]
        else:
            steps += ['source-tests', 'compiled-gate']
        self.config = {'schema': gate.SCHEMA, 'gate': profile,
            'container_name': 'g3r-fixture-once', 'source': gate.source_pin(self.source),
            'dependencies': deps,
            'docker': {'client': str(self.client), 'client_sha256': gate.sha(self.client),
                       'socket': str(self.sockpath), 'root': str(self.docker_root),
                       'image': 'sha256:' + '3' * 64},
            'evidence': {'steps': steps, 'empty_stderr': steps, 'artifact_pins': {}}}
        write(self.prep / 'inside.sh', b'#!/bin/bash\n# sealed, never executed in tests\n')
        self.seal()
        self.locks = tuple(root / name for name in ('heavy.lock', 'docker.lock'))
        self.sleep_calls = []
        self.process = FakeDocker(self)

    def close(self):
        self.sock.close()

    def seal(self):
        write(self.prep / 'config.json', gate.json_bytes(self.config))
        sealpath = self.prep / 'seal.json'
        if sealpath.exists():
            sealpath.unlink()
        self.expected = gate.seal_preparation(self.package)

    def inputs(self):
        return gate.Inputs(self.package, self.expected)

    def sample(self, _inputs):
        self.assert_locks()
        return {'mem_available_kib': gate.MEM_KIB, 'swap_free_kib': gate.SWAP_KIB,
                'scratch_free_bytes': gate.DISK_BYTES, 'docker_free_bytes': gate.DISK_BYTES}

    def assert_locks(self):
        for path in self.locks:
            with path.open('a+') as stream:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                raise AssertionError('shared lock released while custody active')

    def sleep(self, seconds):
        self.assert_locks()
        self.sleep_calls.append(seconds)

    def runner(self, **kwargs):
        return gate.Runner(self.inputs(), process=self.process, locks=self.locks,
                           sample=kwargs.pop('sample', self.sample), sleep=self.sleep,
                           clock=lambda: 1729.0, **kwargs)

    def outputs(self):
        out = self.package / 'out'
        write(out / 'inside.exit', b'0\n')
        for name in self.config['evidence']['steps']:
            write(out / f'steps/{name}.argv', b'exact fixture argv\n')
            write(out / f'steps/{name}.stdout', b'fixture step\n')
            write(out / f'steps/{name}.stderr', b'')
            write(out / f'steps/{name}.exit', b'0\n')
        if self.profile == 'replay':
            for mode in ('normal', 'repeat', 'sanitize'):
                write(out / f'steps/{mode}-run.stdout', trace())
                write(out / f'steps/{mode}-oracle.stdout', b'G3-R replayed P1/G1 independent G3-S oracle PASS\n')
            for mode in ('normal', 'sanitize'):
                write(out / f'build/{mode}/caller', binary(mode == 'sanitize'))
                write(out / f'build/{mode}/caller.ll', b'call @et_g3t_private_replayed_p1_g1_acquire_v1\n')
        else:
            base = out / 'authentic-save'
            for name in ('no-save/program', 'producer/program', 'private-normal/pair-file-reader', 'private-sanitize/pair-file-reader'):
                write(base / name, binary('sanitize' in name))
            write(base / 'producer/program.ll', b'call @et_e1b_public_c2_checkpoint_save_v1\n')
            write(base / 'no-save/program.ll', b'; compiled no-SAVE control\n')
            saved = bytes(make_fixture()[0])
            companion = companion_record(hashlib.sha256(saved).digest())
            write(base / 'roundtrip.c2', saved)
            write(base / 'seed/valid.c2', saved)
            write(base / 'companions/valid.g3r', companion)
            (base / 'companions/link.g3r').symlink_to('valid.g3r')
            os.mkfifo(base / 'companions/fifo.g3r')
            pairs = (gate.digest(saved) + '  /out/authentic-save/roundtrip.c2\n'
                     + gate.digest(companion) + '  /out/authentic-save/companions/valid.g3r\n').encode()
            write(base / 'pair-before-reader.sha256', pairs)
            write(base / 'pair-after-reader.sha256', pairs)
            write(base / 'no-save/output-absence.txt', b'absent after compiled LOAD/release without SAVE\n')
            write(base / 'no-save/run.stdout', b'G3-R NO-SAVE CONTROL PASS\n')
            write(base / 'producer/run.stdout', b'C2 PUBLIC RUNTIME PASS: 10 checks\n')
            write(base / 'provenance.tsv', ('source_head\t' + self.config['source']['head']
                  + '\nsource_tree\t' + self.config['source']['tree'] + '\n').encode())
            for name in ('private-normal/run', 'private-normal/repeat', 'private-sanitize/run'):
                write(base / (name + '.stdout'), b'G3-R companion file read PASS: 29 checks\n')
            for name in ('no-save/run', 'producer/run', 'private-normal/run', 'private-normal/repeat', 'private-sanitize/run'):
                write(base / (name + '.stderr'), b'')


class FakeDocker:
    def __init__(self, fixture):
        self.f = fixture
        self.calls, self.events = [], ['terminal']
        self.run_calls, self.run_code = 0, 0
        self.interrupt, self.cidfile = False, True
        self.mutate = lambda: None
        self.occupied = False
        self.cid = '4' * 64

    def absent(self, stderr):
        stderr.write(('Error response from daemon: No such container: '
                      + self.f.config['container_name'] + '\n').encode())
        return 1

    def row(self, state='exited'):
        return {'Id': self.cid, 'Name': '/' + self.f.config['container_name'],
                'Image': self.f.config['docker']['image'], 'Config': {'Labels': {
                    'codex.source-head': self.f.config['source']['head'],
                    'codex.evidence-kind': 'g3r-' + self.f.profile,
                    'codex.preparation-sha256': self.f.expected}},
                'State': {'Running': state == 'running', 'Status': state, 'ExitCode': 0}}

    def execute(self, argv, stdout, stderr, timeout):
        self.f.assert_locks()
        self.calls.append((argv, timeout))
        assert argv[:5] == [str(self.f.client), '--host', 'unix://' + str(self.f.sockpath),
                            '--config', str(self.f.package / 'out/docker-config')]
        args = argv[5:]
        if args[:2] == ['info', '--format']:
            stdout.write((b'["name=rootless"]\n' if 'SecurityOptions' in args[2]
                          else (str(self.f.docker_root) + '\n').encode()))
        elif args[:2] == ['image', 'inspect']:
            stdout.write(gate.json_bytes([{'Id': self.f.config['docker']['image'],
                                          'Os': 'linux', 'Architecture': 'amd64'}]))
        elif args[:2] == ['container', 'inspect']:
            if self.run_calls == 0:
                if self.occupied:
                    stdout.write(gate.json_bytes([self.row()]))
                else:
                    stdout.write(b'[]\n')  # Measured rootless Docker missing-name response.
                    return self.absent(stderr)
            else:
                event = self.events.pop(0)
                if event == 'error':
                    stderr.write(b'cannot connect to daemon\n')
                    return 125
                if event == 'absent':
                    stdout.write(b'[]\n')
                    return self.absent(stderr)
                row = self.row('running' if event == 'running' else 'exited')
                if event == 'foreign':
                    row['Config']['Labels']['codex.preparation-sha256'] = '5' * 64
                if event == 'other-id':
                    row['Id'] = '6' * 64
                stdout.write(gate.json_bytes([row]))
        elif args[0] == 'run':
            assert timeout is None
            assert argv == self.f.inputs().command()
            self.run_calls += 1
            if self.cidfile:
                write(self.f.package / 'out/container.cid', (self.cid + '\n').encode())
            self.f.outputs()
            self.mutate()
            stdout.write(b'fixture attached Docker output\n')
            if self.interrupt:
                raise gate.ClientInterrupted('fake client signal')
            return self.run_code
        else:
            raise AssertionError('unexpected Docker action: ' + repr(args))
        return 0


class OneUseGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.fixture = Fixture(self.root)
        self.addCleanup(self.fixture.close)
        self.addCleanup(self.tmp.cleanup)

    def test_success_and_complete_output_seal(self):
        f = self.fixture
        result = f.runner().run()
        self.assertEqual(result['status'], 'PASS')
        seal = json.loads((f.package / 'out/seal.json').read_text())
        self.assertEqual(seal['files'], gate.manifest(f.package / 'out', ('seal.json',), special=True))
        self.assertEqual(f.process.run_calls, 1)
        self.assertEqual(json.loads((f.package / 'out/run.argv.json').read_text()), f.inputs().command())
        for args, _ in f.process.calls:
            self.assertNotIn('--rm', args)
            self.assertNotIn('--restart', args)
            self.assertNotIn('kill', args)
            self.assertNotIn('stop', args)
        self.assertIn('--sig-proxy=false', f.inputs().command())

    def test_reused_attempt_does_not_invoke_docker(self):
        f = self.fixture
        f.runner().run()
        calls = len(f.process.calls)
        with self.assertRaisesRegex(gate.Hold, 'stale/reused'):
            f.runner().run()
        self.assertEqual(len(f.process.calls), calls)

    def test_dirty_or_untracked_source_rejected(self):
        f = self.fixture
        for rel in ('fixture.esk', 'untracked.esk'):
            with self.subTest(rel=rel):
                old = (f.source / rel).read_bytes() if (f.source / rel).exists() else None
                write(f.source / rel, b'mutated')
                with self.assertRaises(gate.Hold):
                    f.runner().run()
                if old is None:
                    (f.source / rel).unlink()
                else:
                    write(f.source / rel, old)
        self.assertFalse(f.process.calls)

    def test_seal_and_dependency_tampering_rejected_before_claim(self):
        f = self.fixture
        for path in (f.prep / 'inside.sh', self.root / 'eshkol-build/pin', f.client):
            old = path.read_bytes()
            with self.subTest(path=path):
                write(path, b'tampered')
                with self.assertRaises(gate.Hold):
                    f.runner().run()
                write(path, old)
        self.assertFalse((f.package / 'CLAIM.json').exists())
        self.assertFalse(f.process.calls)

    def test_stale_expected_seal_and_duplicate_json_rejected(self):
        f = self.fixture
        with self.assertRaises(gate.Hold):
            gate.Inputs(f.package, '0' * 64)
        with self.assertRaisesRegex(gate.Hold, 'duplicate JSON'):
            gate.load_json(b'{"x":1,"x":2}')
        with self.assertRaises((OSError, gate.Hold)):
            gate.seal_preparation(f.package)

    def test_symlink_input_leaf_and_ancestor_are_rejected(self):
        f = self.fixture
        target = f.prep / 'inside.sh'
        old = target.read_bytes()
        target.unlink()
        target.symlink_to(f.client)
        with self.assertRaises(gate.Hold):
            f.runner().run()
        target.unlink()
        write(target, old)
        link = self.root / 'alias'
        link.symlink_to(f.package, target_is_directory=True)
        with self.assertRaises(OSError):
            gate.Inputs(link, f.expected)

    def test_mount_and_manifest_path_escapes_rejected(self):
        for value in ('../escape', '/absolute', 'x/../escape', './file', 'x//file'):
            with self.subTest(value=value), self.assertRaises(gate.Hold):
                gate.relative(value)
        f = self.fixture
        f.config['dependencies'][0]['guest'] = '/out'
        f.seal()
        with self.assertRaisesRegex(gate.Hold, 'dependency mount paths'):
            f.inputs()

    def test_resource_floors_checked_while_both_locks_held(self):
        f = self.fixture
        for key in ('mem_available_kib', 'swap_free_kib', 'scratch_free_bytes', 'docker_free_bytes'):
            def low(inputs, key=key):
                data = f.sample(inputs)
                data[key] -= 1
                return data
            with self.subTest(key=key), self.assertRaisesRegex(gate.Hold, key):
                f.runner(sample=low).run()
            self.assertFalse((f.package / 'CLAIM.json').exists())
        self.assertFalse(f.process.calls)

    def test_busy_lock_never_claims(self):
        f = self.fixture
        for path in f.locks:
            with path.open('a+') as held:
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(gate.Hold, 'slot occupied'):
                    f.runner().run()
        self.assertFalse((f.package / 'CLAIM.json').exists())

    def test_occupied_name_never_runs(self):
        f = self.fixture
        f.process.occupied = True
        with self.assertRaisesRegex(gate.Hold, 'occupied'):
            f.runner().run()
        self.assertEqual(f.process.run_calls, 0)
        self.assertEqual(json.loads((f.package / 'out/result.json').read_text())['status'], 'FAIL')

    def test_interrupt_uncertain_inspect_then_terminal_retains_locks(self):
        f = self.fixture
        f.process.interrupt = True
        f.process.events = ['error', 'running', 'terminal']
        with self.assertRaisesRegex(gate.Hold, 'fake client signal'):
            f.runner().run()
        self.assertEqual(f.sleep_calls, [30, 30])
        self.assertEqual(f.process.run_calls, 1)
        self.assertTrue((f.package / 'out/host/run.error').exists())
        self.assertEqual(json.loads((f.package / 'out/custody.json').read_text())['status'], 'owned-terminal')
        self.assertTrue((f.package / 'out/seal.json').exists())

    def test_two_spaced_absences_are_required_and_uncertainty_resets_count(self):
        f = self.fixture
        f.process.run_code = 1
        f.process.events = ['absent', 'error', 'absent', 'absent']
        with self.assertRaises(gate.Hold):
            f.runner().run()
        self.assertEqual(f.sleep_calls, [5, 30, 5])
        self.assertEqual(json.loads((f.package / 'out/custody.json').read_text())['observations'], 2)

    def test_foreign_container_and_changed_cid_do_not_release_lock(self):
        f = self.fixture
        f.process.events = ['foreign', 'running', 'other-id', 'terminal']
        f.runner().run()
        self.assertEqual(f.sleep_calls, [30, 30, 30])
        self.assertEqual(f.process.run_calls, 1)

    def test_missing_cid_cannot_establish_success(self):
        f = self.fixture
        f.process.cidfile = False
        with self.assertRaises(gate.Hold):
            f.runner().run()
        self.assertTrue((f.package / 'out/custody.json').exists())

    def test_guest_failure_incomplete_streams_and_marker_only_are_rejected(self):
        mutations = [
            lambda o: (o / 'inside.exit').write_bytes(b'1\n'),
            lambda o: (o / 'steps/normal-run.exit').unlink(),
            lambda o: (o / 'steps/normal-run.stderr').write_bytes(b'ASan error'),
            lambda o: (o / 'build/sanitize/caller').unlink(),
            lambda o: (o / 'build/normal/caller.ll').write_bytes(b'static oracle control'),
            lambda o: (o / 'steps/sanitize-run.stdout').write_bytes(b'PASS\n'),
        ]
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                f = Fixture(self.root / ('case' + str(index)))
                self.addCleanup(f.close)
                f.process.mutate = lambda f=f, m=mutation: m(f.package / 'out')
                with self.assertRaises(gate.Hold):
                    f.runner().run()
                self.assertEqual(json.loads((f.package / 'out/result.json').read_text())['status'], 'FAIL')
                self.assertTrue((f.package / 'out/seal.json').exists())

    def test_output_symlink_escape_is_sealed_as_failure_without_reading_target(self):
        f = self.fixture
        def mutate():
            p = f.package / 'out/build/normal/caller'
            p.unlink()
            p.symlink_to('/etc/passwd')
        f.process.mutate = mutate
        with self.assertRaisesRegex(gate.Hold, 'output link'):
            f.runner().run()
        seal = json.loads((f.package / 'out/seal.json').read_text())
        self.assertEqual(seal['files']['build/normal/caller'], 'symlink:/etc/passwd')

    def test_artifact_pin_and_host_launch_argv_cannot_drift(self):
        for index, mutate_argv in enumerate((False, True)):
            f = Fixture(self.root / ('identity' + str(index)))
            self.addCleanup(f.close)
            if mutate_argv:
                f.process.mutate = lambda f=f: write(f.package / 'out/run.argv.json', b'[]\n')
            else:
                f.config['evidence']['artifact_pins'] = {'build/normal/caller': '0' * 64}
                f.seal()
            with self.subTest(argv=mutate_argv), self.assertRaises(gate.Hold):
                f.runner().run()

    def test_postflight_source_mutation_is_not_admitted(self):
        f = self.fixture
        f.process.mutate = lambda: write(f.source / 'fixture.esk', b'dirty after compile')
        with self.assertRaisesRegex(gate.Hold, 'source manifest'):
            f.runner().run()

    def test_actual_trace_oracle_rejects_wrong_sample_and_rng(self):
        from tests.g3r.replay_gate_oracle import check_trace
        check_trace(trace())
        for data in (trace().replace(b'POST 217', b'POST 218'),
                     trace().replace(b'0 0\nG3-R', b'0 1\nG3-R'),
                     trace().replace(b'PRE categorical', b'PRE greedy')):
            with self.subTest(data=data[-70:]), self.assertRaises(ValueError):
                check_trace(data)

    def test_save_profile_validates_authentic_pair_and_expected_negative_nodes(self):
        f = Fixture(self.root / 'save', 'save')
        self.addCleanup(f.close)
        f.runner().run()
        seal = json.loads((f.package / 'out/seal.json').read_text())
        self.assertEqual(seal['files']['authentic-save/companions/link.g3r'], 'symlink:valid.g3r')
        self.assertEqual(seal['files']['authentic-save/companions/fifo.g3r'], 'mode:' + str(__import__('stat').S_IFIFO))

    def test_save_pair_and_producer_identity_mutations_fail(self):
        mutations = [
            lambda b: write(b / 'producer/program.ll', b'no SAVE call'),
            lambda b: write(b / 'no-save/program.ll', b'et_e1b_public_c2_checkpoint_save_v1'),
            lambda b: write(b / 'roundtrip.c2', b'corrupted'),
            lambda b: write(b / 'pair-after-reader.sha256', b'changed'),
            lambda b: write(b / 'private-sanitize/run.stdout', b'static PASS'),
            lambda b: write(b / 'provenance.tsv', b'source_head\twrong\n'),
            lambda b: (b / 'companions/link.g3r').unlink() or (b / 'companions/link.g3r').symlink_to('/etc/passwd'),
        ]
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                f = Fixture(self.root / ('save-negative' + str(index)), 'save')
                self.addCleanup(f.close)
                f.process.mutate = lambda f=f, m=mutation: m(f.package / 'out/authentic-save')
                with self.assertRaises(gate.Hold):
                    f.runner().run()

    def test_wrong_save_special_name_or_type_is_rejected(self):
        def wrong_name(base):
            (base / 'companions/link.g3r').rename(base / 'companions/other.g3r')
        def wrong_type(base):
            (base / 'companions/fifo.g3r').unlink()
            (base / 'companions/fifo.g3r').symlink_to('valid.g3r')
        for index, mutation in enumerate((wrong_name, wrong_type)):
            f = Fixture(self.root / ('special' + str(index)), 'save')
            self.addCleanup(f.close)
            f.process.mutate = lambda f=f, m=mutation: m(f.package / 'out/authentic-save')
            with self.subTest(index=index), self.assertRaisesRegex(gate.Hold, 'output link'):
                f.runner().run()

    def test_custody_log_failure_never_releases_an_uncertain_container(self):
        f = self.fixture
        f.process.events = ['error', 'running', 'terminal']
        original = gate.write_new
        def fail_note(path, data):
            if path.name.startswith('custody-error-'):
                raise OSError('fixture logging unavailable')
            return original(path, data)
        with patch.object(gate, 'write_new', side_effect=fail_note):
            f.runner().run()
        self.assertEqual(f.sleep_calls, [30, 30])

    def test_ambient_docker_endpoint_variables_do_not_change_vector(self):
        f = self.fixture
        before = f.inputs().command()
        with patch.dict(os.environ, {'DOCKER_HOST': 'tcp://wrong', 'DOCKER_CONTEXT': 'wrong'}):
            self.assertEqual(f.inputs().command(), before)
            f.runner().run()

    def test_measured_docker_absence_dialect_and_near_misses(self):
        name = 'g3r-fixture-once'
        error = ('Error response from daemon: No such container: ' + name + '\n').encode()
        for data in (b'', b'[]\n'):
            self.assertTrue(gate.confirmed_absence(1, data, error, name))
        for code, data, err in (
            (0, b'[]\n', error), (2, b'[]\n', error), (1, b'{}\n', error),
            (1, b'[{}]\n', error), (1, b'not-json', error),
            (1, b'[]\n', b'cannot connect to daemon\n'),
            (1, b'[]\n', error.replace(name.encode(), b'other-name')),
            (1, b'[]\n', error + b'additional daemon error\n'),
        ):
            with self.subTest(code=code, data=data, err=err):
                self.assertFalse(gate.confirmed_absence(code, data, err, name))

    def test_real_process_adapter_scrubs_endpoint_and_only_reaps_client_group(self):
        from unittest.mock import Mock
        proc = Mock(pid=1729, returncode=-signal.SIGKILL)
        proc.wait.side_effect = [gate.ClientInterrupted('fixture signal'),
                                subprocess.TimeoutExpired('client', 5), -signal.SIGKILL]
        with patch.dict(os.environ, {'DOCKER_HOST': 'tcp://wrong'}), \
                patch.object(gate.subprocess, 'Popen', return_value=proc) as popen, \
                patch.object(gate.os, 'killpg') as killpg:
            with self.assertRaises(gate.ClientInterrupted) as raised:
                gate.Process().execute(['pinned-docker', 'run'], None, None, None)
        self.assertEqual(raised.exception.returncode, -signal.SIGKILL)
        self.assertNotIn('DOCKER_HOST', popen.call_args.kwargs['env'])
        self.assertEqual(killpg.call_args_list, [__import__('unittest').mock.call(1729, signal.SIGTERM),
                                               __import__('unittest').mock.call(1729, signal.SIGKILL)])


if __name__ == '__main__':
    unittest.main()
