#!/usr/bin/python3
"""Synthetic-only runner tests; no F/G product path is opened."""
import hashlib
import fcntl
import importlib.util
import json
import copy
import contextlib
import io
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock
import subprocess

here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('runner_under_test', here / 'RUNNER.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
fixture_path = r.PARSER / 'TEST_INVENTORY.py'
fixture_spec = importlib.util.spec_from_file_location('synthetic_fixtures', fixture_path)
fixtures = importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(fixtures)
parser_spec = importlib.util.spec_from_file_location('sealed_parser_for_tests', r.PARSER / 'INVENTORY.py')
parser = importlib.util.module_from_spec(parser_spec)
parser_spec.loader.exec_module(parser)


class RunnerTest(unittest.TestCase):
    def test_caller_owned_completion_modes_are_reversible(self):
        # Runner J's final chmod does not authenticate caller-owned output.
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'caller-owned-output'
            output.mkdir(mode=0o700)
            result = output / 'RESULT.json'
            result.write_text('original\n')
            result.chmod(0o444)
            output.chmod(0o555)
            output.chmod(0o700)
            result.chmod(0o600)
            result.write_text('forged\n')
            self.assertEqual(result.read_text(), 'forged\n')

    def test_plain_python_exec_resets_nondumpable_state(self):
        # An in-launcher PR_SET_DUMPABLE(0) does not protect inherited FD3
        # after the current ordinary-Python exec. This is a kernel contract
        # diagnostic, not a claim that the synthetic runner is installed.
        child = ('import ctypes, os, sys; '
                 'prctl = ctypes.CDLL(None).prctl; '
                 'assert prctl(4, 0, 0, 0, 0) == 0; '
                 'assert prctl(3, 0, 0, 0, 0) == 0; '
                 'os.execv(sys.executable, [sys.executable, "-I", "-S", "-c", '
                 '"import ctypes; print(ctypes.CDLL(None).prctl(3, 0, 0, 0, 0))"])')
        result = subprocess.run(['/usr/bin/python3', '-I', '-S', '-c', child],
                                capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), '1')

    def test_child_handoff_parent_output_and_stale_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            output = base / (r.STEM + 'inventory-run-synthetic')
            output.mkdir(mode=0o700)
            st = output.stat()
            binding = 'b' * 64
            parent = os.getppid()
            start = r.process_starttime(parent)

            def handoff(pid=parent, started=start, ino=st.st_ino):
                raw = (f'eshkol-off-inventory-handoff-v2\n{binding}\n'
                       f'{os.getuid()}\n{pid}\n{started}\n{st.st_dev}\n'
                       f'{ino}\n{"a" * 32}\n').encode()
                fd = os.memfd_create('synthetic-child-handoff', os.MFD_ALLOW_SEALING)
                os.fchmod(fd, 0o600)
                os.write(fd, raw)
                fcntl.fcntl(fd, fcntl.F_ADD_SEALS, fcntl.F_SEAL_WRITE |
                            fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
                return fd

            with mock.patch.object(r, 'BASE', base), mock.patch.object(
                    r, 'protected_handoff_stat', return_value=True):
                for fd, reason in ((handoff(pid=parent + 1), 'parent drift'),
                                   (handoff(started='0'), 'parent drift'),
                                   (handoff(ino=0), 'output inode drift')):
                    try:
                        with self.assertRaisesRegex(r.Hold, reason):
                            r.consume_child_handoff(output, binding, fd)
                    finally:
                        os.close(fd)
                fd = handoff()
                duplicate = os.dup(fd)
                self.assertEqual(r.consume_child_handoff(output, binding, fd),
                                 (st.st_dev, st.st_ino))
                with self.assertRaises(OSError):
                    os.fstat(fd)
                try:
                    # A copy in an unrelated caller cannot claim the live run parent.
                    with mock.patch.object(r.os, 'getppid', return_value=parent + 1):
                        with self.assertRaisesRegex(r.Hold, 'parent drift'):
                            r.consume_child_handoff(output, binding, duplicate)
                finally:
                    os.close(duplicate)

    def test_direct_child_requires_fd_before_release_or_product(self):
        argv = ['/usr/bin/python3', '-I', '-S', str(here / 'RUNNER.py'), '--child',
                'fb', 'object', 'unit.o', '/definitely/missing.o',
                '/definitely/missing.receipt.json', '/definitely/missing-release.json',
                'b' * 64, '3']
        result = subprocess.run(argv, capture_output=True, text=True, close_fds=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('Bad file descriptor', result.stderr)
        self.assertNotIn('missing-release', result.stderr)

    def test_inherited_fd3_matches_live_parent_in_real_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            output = base / (r.STEM + 'inventory-run-synthetic')
            output.mkdir(mode=0o700)
            st = output.stat()
            binding = 'b' * 64
            raw = (f'eshkol-off-inventory-handoff-v2\n{binding}\n'
                   f'{os.getuid()}\n{os.getpid()}\n{r.process_starttime()}\n'
                   f'{st.st_dev}\n{st.st_ino}\n{"a" * 32}\n').encode()
            fd = os.memfd_create('synthetic-fd3', os.MFD_ALLOW_SEALING)
            try:
                os.fchmod(fd, 0o600)
                os.write(fd, raw)
                fcntl.fcntl(fd, fcntl.F_ADD_SEALS, fcntl.F_SEAL_WRITE |
                            fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
                code = ("import importlib.util,sys; from pathlib import Path; "
                        f"p=Path({str(here / 'RUNNER.py')!r}); "
                        "s=importlib.util.spec_from_file_location('synthetic_runner',p); "
                        "r=importlib.util.module_from_spec(s); s.loader.exec_module(r); "
                        f"r.BASE=Path({str(base)!r}); "
                        "r.protected_handoff_stat=lambda st: True; "
                        f"print(r.consume_child_handoff(Path({str(output)!r}),"
                        f"{binding!r},3))")
                result = subprocess.run(['/usr/bin/python3', '-I', '-S', '-c', code],
                                        pass_fds=(fd,), preexec_fn=lambda: os.dup2(fd, 3),
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(str(st.st_ino), result.stdout)
                # A copied descriptor in a caller-created grandchild has a
                # different live parent and cannot reuse the launch record.
                nested = ("import subprocess; c=" + repr(code) + "; "
                          "p=subprocess.run(['/usr/bin/python3','-I','-S','-c',c],"
                          "pass_fds=(3,),capture_output=True,text=True); "
                          "print(p.returncode); print(p.stderr)")
                replay = subprocess.run(['/usr/bin/python3', '-I', '-S', '-c', nested],
                                        pass_fds=(fd,), preexec_fn=lambda: os.dup2(fd, 3),
                                        capture_output=True, text=True)
                self.assertEqual(replay.returncode, 0, replay.stderr)
                self.assertIn('child handoff parent drift or replay', replay.stdout)
            finally:
                os.close(fd)

    def test_direct_child_admitted_synthetic_product_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            run_root = base / 'synthetic-run'
            (run_root / 'build').mkdir(parents=True)
            raw = fixtures.elf64()
            product_path = run_root / 'build/unit.o'
            product_path.write_bytes(raw)
            product = {'path': 'unit.o', 'bytes': len(raw), 'sha256': r.sha_bytes(raw)}
            seal = run_root / 'SHA256SUMS-RUN'
            seal.write_text(f'{product["sha256"]}  build/unit.o\n')
            output = base / (r.STEM + 'inventory-run-synthetic')
            output.mkdir(mode=0o700)
            receipt = output / 'fb-unit.o.receipt.json'
            argv = ['RUNNER.py', '--child', 'fb', 'object', 'unit.o',
                    str(product_path), str(receipt), str(base / 'RELEASE.json'),
                    'b' * 64, '3']
            sides = {'fb': ([product],), 'gb': ([],)}
            with mock.patch.object(r, 'BASE', base), mock.patch.dict(
                    r.RUN, {'fb': run_root}), mock.patch.dict(
                    r.ROOT_SHA, {'fb': r.sha_file(seal)}), mock.patch.object(
                    r, 'consume_child_handoff') as handoff, mock.patch.object(
                    r, 'review_gate') as review, mock.patch.object(
                    r, 'authenticate_metadata', return_value=(None, [
                        {'path': 'unit.o', 'kind': 'object'}], None, sides)), mock.patch.object(
                    r, 'verify_tools'), mock.patch.object(r, 'child_parse') as parse:
                r.main(argv)
                handoff.assert_called_once_with(output, 'b' * 64, 3)
                review.assert_called_once()
                parse.assert_called_once()
                self.assertEqual(parse.call_args.args[3]['sha256'], product['sha256'])
                self.assertEqual(parse.call_args.args[3]['kind'], 'object')

    def test_precreated_output_sealed_handoff_binds_inode_and_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            output = base / (r.STEM + 'inventory-run-synthetic')
            output.mkdir(mode=0o700)
            binding = 'b' * 64
            st = output.stat()
            def handoff(ino, pid=os.getpid(), seal=True):
                record = (f'eshkol-off-inventory-handoff-v2\n{binding}\n'
                          f'{os.getuid()}\n{pid}\n{r.process_starttime()}\n'
                          f'{st.st_dev}\n{ino}\n{"a" * 32}\n').encode()
                fd = os.memfd_create('synthetic-handoff', os.MFD_ALLOW_SEALING)
                os.fchmod(fd, 0o600)
                os.write(fd, record)
                if seal:
                    flags = (fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW |
                             fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
                    fcntl.fcntl(fd, fcntl.F_ADD_SEALS, flags)
                return fd
            with mock.patch.object(r, 'BASE', base):
                fd = handoff(st.st_ino)
                try:
                    with self.assertRaisesRegex(r.Hold, 'not protected'):
                        r.consume_launcher_handoff(output, binding, fd)
                    with mock.patch.object(r, 'protected_handoff_stat', return_value=True):
                        output.joinpath('preexisting').write_text('x')
                        with self.assertRaisesRegex(r.Hold, 'freshness'):
                            r.consume_launcher_handoff(output, binding, fd)
                        output.joinpath('preexisting').unlink()
                        wrong = handoff(0)
                        with self.assertRaisesRegex(r.Hold, 'inode drift'):
                            r.consume_launcher_handoff(output, binding, wrong)
                        os.close(wrong)
                        wrong_pid = handoff(st.st_ino, os.getpid() + 1)
                        with self.assertRaisesRegex(r.Hold, 'record drift'):
                            r.consume_launcher_handoff(output, binding, wrong_pid)
                        os.close(wrong_pid)
                        self.assertEqual(r.consume_launcher_handoff(output, binding, fd),
                                         (st.st_dev, st.st_ino))
                finally:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                fd = handoff(st.st_ino, seal=False)
                try:
                    with mock.patch.object(r, 'protected_handoff_stat', return_value=True):
                        with self.assertRaisesRegex(r.Hold, 'not sealed'):
                            r.consume_launcher_handoff(output, binding, fd)
                finally:
                    os.close(fd)

    def test_absent_zip_preflight_rejects_mutable_archive_and_symlink(self):
        r.preflight_absent_stdlib_zip()
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / 'python314.zip'
            archive.write_bytes(b'synthetic archive only')
            with self.assertRaisesRegex(RuntimeError, 'search entry present'):
                r.preflight_absent_stdlib_zip(str(archive))
            archive.unlink()
            archive.symlink_to(Path(tmp) / 'mutable.zip')
            with self.assertRaisesRegex(RuntimeError, 'search entry present'):
                r.preflight_absent_stdlib_zip(str(archive))
            archive.unlink()
            with self.assertRaisesRegex(RuntimeError, 'parent mutable'):
                r.preflight_absent_stdlib_zip(str(archive))

    def parse_synthetic(self, kind, input_path, receipt_path, product):
        base_fd = os.open(receipt_path.parent.parent, os.O_RDONLY | os.O_DIRECTORY)
        output_fd = os.open(receipt_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            return r.child_parse(kind, input_path, receipt_path, product,
                                 (base_fd, output_fd))
        finally:
            os.close(output_fd)
            os.close(base_fd)

    def test_stdlib_requires_root_owned_nonwritable_path(self):
        r.trusted_root_owned_path(Path('/usr/lib/python3.14/sysconfig/__init__.py'))
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'sysconfig.py'
            source.write_text('VALUE = 1\n')
            with self.assertRaisesRegex(r.Hold, 'writable outside root'):
                r.trusted_root_owned_path(source)

    def test_isolated_import_resolution_rejects_unsealed_plan_path(self):
        code = ("import importlib.util,sys; from pathlib import Path; "
                f"p=Path({str(here / 'RUNNER.py')!r}); "
                "s=importlib.util.spec_from_file_location('isolated_runner',p); "
                "r=importlib.util.module_from_spec(s); s.loader.exec_module(r); "
                "r.freeze_import_resolution(); "
                "sys.path.insert(0,'/synthetic/unsealed/plan'); "
                "\ntry: r.freeze_import_resolution()\n"
                "except r.Hold: print('unsealed path rejected')\n"
                "else: raise AssertionError('unsealed path admitted')")
        result = subprocess.run(['/usr/bin/python3', '-I', '-S', '-c', code],
                                check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), 'unsealed path rejected')

    def test_import_path_pin_cannot_be_redefined_to_mutable_root(self):
        code = ("import importlib.util,sys; from pathlib import Path; "
                f"p=Path({str(here / 'RUNNER.py')!r}); "
                "s=importlib.util.spec_from_file_location('isolated_runner',p); "
                "r=importlib.util.module_from_spec(s); s.loader.exec_module(r); "
                "r.SAFE_IMPORT_PATH=('/tmp',); sys.path[:]=['/tmp']; "
                "\ntry: r.freeze_import_resolution()\n"
                "except r.Hold: print('mutable root rejected')\n"
                "else: raise AssertionError('mutable root admitted')")
        result = subprocess.run(['/usr/bin/python3', '-I', '-S', '-c', code],
                                check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), 'mutable root rejected')

    def test_pinned_run_snapshot_prevents_transient_parse_hash_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'build').mkdir()
            product = root / 'build/unit.o'
            product.write_bytes(b'synthetic only')
            result_path = root / 'RESULT.json'
            good = b'{"products":[{"path":"unit.o","bytes":14}]}\n'
            bad = b'{"products":[{"path":"unit.o","bytes":999}]}\n'
            result_path.write_bytes(good)
            seal = root / 'SHA256SUMS-RUN'
            seal.write_text(f'{r.sha_file(product)}  build/unit.o\n'
                            f'{r.sha_bytes(good)}  RESULT.json\n')
            with mock.patch.dict(r.RUN, {'fb': root}), mock.patch.dict(
                    r.ROOT_SHA, {'fb': r.sha_file(seal)}):
                names, snapshot = r.snapshot_run_metadata('fb')
                self.assertEqual(names, {'build/unit.o'})
                with r.PinnedRunReads(snapshot):
                    result_path.write_bytes(bad)
                    parsed = json.loads(result_path.read_text())
                    hashed = r.sha_bytes(result_path.read_bytes())
                    self.assertEqual(parsed['products'][0]['bytes'], 14)
                    self.assertEqual(hashed, r.sha_bytes(good))
                self.assertEqual(result_path.read_bytes(), bad)
                with self.assertRaisesRegex(r.Hold, 'snapshot drift'):
                    r.snapshot_run_metadata('fb')
                with result_path.open('wb') as stream:
                    stream.truncate(256 * 1024 * 1024 + 1)
                with self.assertRaisesRegex(r.Hold, 'snapshot cap'):
                    r.snapshot_run_metadata('fb')

    def test_child_receipt_owned_fresh_output_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            output = base / (r.STEM + 'inventory-run-synthetic')
            output.mkdir(mode=0o700)
            receipt = output / 'fb-unit.o.receipt.json'
            with mock.patch.object(r, 'BASE', base):
                fds = r.admit_child_receipt('fb', 'unit.o', receipt)
                for fd in fds:
                    os.close(fd)
                with self.assertRaisesRegex(r.Hold, 'not canonical'):
                    r.admit_child_receipt('fb', 'unit.o', output / 'other.json')
                with self.assertRaisesRegex(r.Hold, 'outside owned'):
                    r.admit_child_receipt('fb', 'unit.o', base / 'outside' /
                                          'fb-unit.o.receipt.json')
                receipt.write_text('existing')
                with self.assertRaisesRegex(r.Hold, 'not fresh'):
                    r.admit_child_receipt('fb', 'unit.o', receipt)
                receipt.unlink()
                receipt.symlink_to(base / 'missing')
                with self.assertRaisesRegex(r.Hold, 'not fresh'):
                    r.admit_child_receipt('fb', 'unit.o', receipt)
                receipt.unlink()
                output.chmod(0o755)
                with self.assertRaisesRegex(r.Hold, 'not fresh'):
                    r.admit_child_receipt('fb', 'unit.o', receipt)

    def test_receipt_parent_rename_symlink_cannot_redirect(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            output = base / (r.STEM + 'inventory-run-synthetic')
            output.mkdir(mode=0o700)
            outside = base / 'outside'
            outside.mkdir()
            receipt = output / 'fb-unit.o.receipt.json'
            with mock.patch.object(r, 'BASE', base):
                fds = r.admit_child_receipt('fb', 'unit.o', receipt)
            moved = base / 'moved-output'
            output.rename(moved)
            output.symlink_to(outside, target_is_directory=True)
            try:
                with self.assertRaisesRegex(r.Hold, 'directory moved'):
                    r.write_child_receipt(fds, receipt, {'synthetic': True})
                self.assertFalse((outside / receipt.name).exists())
                self.assertFalse((moved / receipt.name).exists())
            finally:
                for fd in fds:
                    os.close(fd)

    def test_receipt_parent_swap_at_open_unlinks_pinned_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            output = base / (r.STEM + 'inventory-run-synthetic')
            output.mkdir(mode=0o700)
            outside = base / 'outside'
            outside.mkdir()
            receipt = output / 'fb-unit.o.receipt.json'
            with mock.patch.object(r, 'BASE', base):
                fds = r.admit_child_receipt('fb', 'unit.o', receipt)
            moved = base / 'moved-output'
            original_open = os.open
            def swap_at_open(path, flags, *args, **kwargs):
                if path == receipt.name and kwargs.get('dir_fd') == fds[1]:
                    output.rename(moved)
                    output.symlink_to(outside, target_is_directory=True)
                return original_open(path, flags, *args, **kwargs)
            try:
                with mock.patch.object(r.os, 'open', side_effect=swap_at_open):
                    with self.assertRaisesRegex(r.Hold, 'directory moved'):
                        r.write_child_receipt(fds, receipt, {'synthetic': True})
                self.assertFalse((outside / receipt.name).exists())
                self.assertFalse((moved / receipt.name).exists())
            finally:
                for fd in fds:
                    os.close(fd)

    def test_direct_child_admission_rejects_side_kind_path_size_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'build').mkdir()
            raw = fixtures.elf64()
            input_path = root / 'build/unit.o'
            input_path.write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            seal = root / 'SHA256SUMS-RUN'
            seal.write_text(f'{digest}  build/unit.o\n')
            product = {'path': 'unit.o', 'bytes': len(raw), 'sha256': digest}
            pairs = [{'path': 'unit.o', 'kind': 'object'}]
            sides = {'fb': ([product],), 'gb': ([],)}
            with mock.patch.dict(r.RUN, {'fb': root}), mock.patch.dict(
                    r.ROOT_SHA, {'fb': r.sha_file(seal)}):
                self.assertEqual(r.admit_child_product('fb', 'object', 'unit.o',
                                                       input_path, pairs, sides),
                                 {**product, 'side': 'fb', 'kind': 'object'})
                for side, kind, path, selected in (
                    ('wrong', 'object', 'unit.o', input_path),
                    ('fb', 'archive', 'unit.o', input_path),
                    ('fb', 'object', '../unit.o', input_path),
                    ('fb', 'object', 'unit.o', root / 'other.o'),
                ):
                    with self.subTest(side=side, kind=kind, path=path), self.assertRaises(r.Hold):
                        r.admit_child_product(side, kind, path, selected, pairs, sides)
                input_path.write_bytes(raw + b'x')
                with self.assertRaisesRegex(r.Hold, 'size drift'):
                    r.admit_child_product('fb', 'object', 'unit.o', input_path, pairs, sides)
                input_path.write_bytes(raw)
                bad = {'fb': ([{**product, 'sha256': '0' * 64}],), 'gb': ([],)}
                with self.assertRaisesRegex(r.Hold, 'hash map drift'):
                    r.admit_child_product('fb', 'object', 'unit.o', input_path, pairs, bad)
                seal.write_text('0' * 64 + '  build/unit.o\n')
                with self.assertRaisesRegex(r.Hold, 'RUN root drift'):
                    r.admit_child_product('fb', 'object', 'unit.o', input_path, pairs, sides)

    def test_direct_child_entry_never_parses_unadmitted_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pairs = [{'path': 'unit.o', 'kind': 'object'}]
            sides = {'fb': ([{'path': 'unit.o', 'bytes': 1, 'sha256': '0' * 64}],),
                     'gb': ([],)}
            argv = ['RUNNER.py', '--child', 'fb', 'object', 'unit.o',
                    str(root / 'arbitrary.o'), str(root / 'receipt.json'),
                    str(root / 'RELEASE.json'), 'a' * 64, '3']
            with mock.patch.object(r, 'consume_child_handoff'), mock.patch.object(
                    r, 'review_gate'), mock.patch.object(
                    r, 'authenticate_metadata', return_value=(None, pairs, None, sides)), mock.patch.object(
                    r, 'verify_tools'), mock.patch.object(r, 'child_parse') as parse:
                with self.assertRaisesRegex(r.Hold, 'not canonical'):
                    r.main(argv)
                parse.assert_not_called()

    def test_product_hash_rejected_before_parser_invocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = fixtures.elf64()
            source = root / 'unit.o'
            source.write_bytes(raw)
            fake = type('Parser', (), {'MAX_INPUT': 160 * 1024 * 1024,
                                       'parse_elf': mock.Mock(), 'parse_archive': mock.Mock()})()
            with mock.patch.object(r, 'child_limits'), mock.patch.object(
                    r, 'check_limits'), mock.patch.object(r, 'check_imports'), mock.patch.object(
                    r, 'load_pinned', return_value=fake):
                with self.assertRaisesRegex(r.Hold, 'hash drift before parse'):
                    self.parse_synthetic('object', source, root / 'receipt.json',
                                  {'side': 'fb', 'path': 'unit.o',
                                   'bytes': len(raw), 'sha256': '0' * 64})
            fake.parse_elf.assert_not_called()
            self.assertFalse((root / 'receipt.json').exists())

    def test_pinned_source_executes_authenticated_bytes_after_swap(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'sealed_source.py'
            source.write_text("MARK = 'sealed'\n")
            digest = r.sha_file(source)
            original = Path.read_bytes
            def swapped(path):
                raw = original(path)
                if path == source:
                    source.write_text("MARK = 'drifted'\n")
                return raw
            with mock.patch.object(Path, 'read_bytes', swapped):
                loaded = r.load_pinned(source, 'synthetic_sealed_source', digest)
            self.assertEqual(loaded.MARK, 'sealed')
            self.assertEqual(source.read_text(), "MARK = 'drifted'\n")
            with self.assertRaisesRegex(r.Hold, 'Python source drift'):
                r.load_pinned(source, 'synthetic_changed_source', digest)

    def test_run_metadata_rechecked_without_opening_product(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'build').mkdir()
            product = root / 'build/unit.o'
            product.write_bytes(b'synthetic product')
            metadata = root / 'RESULT.json'
            metadata.write_text('{"status":"complete"}\n')
            seal = root / 'SHA256SUMS-RUN'
            seal.write_text(f'{r.sha_file(product)}  build/unit.o\n'
                            f'{r.sha_file(metadata)}  RESULT.json\n')
            with mock.patch.dict(r.RUN, {'fb': root}), mock.patch.dict(
                    r.ROOT_SHA, {'fb': r.sha_file(seal)}):
                original = r.sha_file
                def no_product_read(path):
                    if Path(path) == product:
                        raise AssertionError('product opened during metadata admission')
                    return original(path)
                with mock.patch.object(r, 'sha_file', side_effect=no_product_read):
                    r.authenticate_run_metadata('fb', {'build/unit.o'})
                self.assertEqual(r.read_run_json('fb', 'RESULT.json'), {'status': 'complete'})
                metadata.write_text('{"status":"changed"}\n')
                with self.assertRaisesRegex(r.Hold, 'RUN metadata drift'):
                    r.authenticate_run_metadata('fb', {'build/unit.o'})
                with self.assertRaisesRegex(r.Hold, 'sealed JSON drift'):
                    r.read_run_json('fb', 'RESULT.json')

    def test_external_binding_and_exact_release_closure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner_root = root / 'runner'
            release_root = root / 'release'
            runner_root.mkdir()
            release_root.mkdir()
            source = runner_root / 'RUNNER.py'
            source.write_text('synthetic reviewed source\n')
            manifest = runner_root / 'MANIFEST.json'
            manifest.write_text(json.dumps({
                'schema': 'eshkol-off-inventory-runner-v9',
                'interpreter': r.identity(sys.executable),
                'python_version': sys.version,
                'runtime_imports': {}, 'linked_libraries': {},
            }))
            runner_seal = runner_root / 'SHA256SUMS-RUNNER'
            runner_seal.write_text(f'{r.sha_file(source)}  RUNNER.py\n'
                                   f'{r.sha_file(manifest)}  MANIFEST.json\n')
            exception = release_root / 'EXCEPTIONS.json'
            exception.write_text('[]\n')
            exception_seal = release_root / 'SHA256SUMS-EXCEPTIONS'
            exception_seal.write_text(f'{r.sha_file(exception)}  EXCEPTIONS.json\n')
            release_path = release_root / 'RELEASE.json'
            release = {'schema': 'eshkol-off-inventory-release-v3',
                       'runner_manifest_sha256': r.sha_file(runner_seal),
                       'exception_file_sha256': r.sha_file(exception)}
            release_path.write_text(json.dumps(release))
            binding = {'schema': 'eshkol-off-inventory-binding-v1',
                       'runner_manifest_sha256': r.sha_file(runner_seal),
                       'release_sha256': r.sha_file(release_path),
                       'exception_manifest_sha256': r.sha_file(exception_seal),
                       'exception_file_sha256': r.sha_file(exception)}
            authority = {}
            for role in ('independent', 'root'):
                folder = root / role
                folder.mkdir()
                decision = folder / 'RELEASE.json'
                decision.write_text(json.dumps({
                    'role': role, 'verdict': 'RELEASE',
                    **{key: binding[key] for key in (
                        'runner_manifest_sha256', 'release_sha256',
                        'exception_manifest_sha256', 'exception_file_sha256')},
                }))
                seal = folder / 'SHA256SUMS-REVIEW'
                seal.write_text(f'{r.sha_file(decision)}  RELEASE.json\n')
                authority[role] = seal
                binding[f'{role}_review_seal_sha256'] = r.sha_file(seal)
            binding_path = root / 'PINS.json'
            binding_path.write_text(json.dumps(binding))
            trusted_sha = r.sha_file(binding_path)
            with mock.patch.object(r, 'freeze_import_resolution'), mock.patch.object(
                    r, 'RUNNER_ROOT', runner_root), mock.patch.object(
                    r, 'RELEASE_PATH', release_path), mock.patch.object(
                    r, 'EXCEPTION_SEAL', exception_seal), mock.patch.object(
                    r, 'BINDING_PATH', binding_path), mock.patch.object(
                    r, 'REVIEW_AUTHORITY', authority):
                self.assertEqual(r.review_gate(release_path, trusted_sha), release)
                with self.assertRaisesRegex(r.Hold, 'binding digest absent'):
                    r.review_gate(release_path, None)
                with self.assertRaisesRegex(r.Hold, 'binding digest drift'):
                    r.review_gate(release_path, '0' * 64)
                with mock.patch.object(r, 'EXCEPTION_SEAL', root / 'missing-exception-seal'):
                    with self.assertRaises((r.Hold, FileNotFoundError)):
                        r.review_gate(release_path, trusted_sha)
                original = exception.read_text()
                exception.write_text('[] \n')
                with self.assertRaisesRegex(r.Hold, 'sealed member drift'):
                    r.review_gate(release_path, trusted_sha)
                exception.write_text(original)
                original = source.read_text()
                source.write_text('drifted source\n')
                with self.assertRaisesRegex(r.Hold, 'sealed member drift'):
                    r.review_gate(release_path, trusted_sha)
                source.write_text(original)
                decision = root / 'root/RELEASE.json'
                original = decision.read_text()
                decision.write_text(original.replace('RELEASE', 'HOLD'))
                with self.assertRaisesRegex(r.Hold, 'sealed member drift'):
                    r.review_gate(release_path, trusted_sha)
                decision.write_text(original)
                original = release_path.read_text()
                release_path.write_text(original + '\n')
                with self.assertRaisesRegex(r.Hold, 'release file differs'):
                    r.review_gate(release_path, trusted_sha)
                release_path.write_text(original)
                binding_path.write_text(json.dumps({**binding, 'self_asserted': trusted_sha}))
                with self.assertRaisesRegex(r.Hold, 'binding digest drift'):
                    r.review_gate(release_path, trusted_sha)
                with self.assertRaisesRegex(r.Hold, 'binding schema drift'):
                    r.review_gate(release_path, r.sha_file(binding_path))
                binding_path.write_text(json.dumps({**binding,
                    'root_review_seal_sha256': binding['independent_review_seal_sha256']}))
                with self.assertRaisesRegex(r.Hold, 'share one seal digest'):
                    r.review_gate(release_path, r.sha_file(binding_path))

    def test_relative_path_rejects_escape(self):
        for path in ('../x', '/tmp/x', 'x/../y'):
            with self.subTest(path=path), self.assertRaises(r.Hold):
                r.relative(path)

    def test_archive_order_and_negative(self):
        path = 'libone.a'
        member = 'CMakeFiles/unit.dir/unit.o'
        cmd = ['/usr/bin/bash', '-c',
               f': && /usr/bin/ar qc {path}  {member} && /usr/bin/ranlib {path} && :']
        self.assertEqual(r.archive_order(cmd, path, {member}), [member])
        with self.assertRaises(r.Hold):
            r.archive_order(cmd, path, {})
        with self.assertRaises(r.Hold):
            r.archive_order(cmd, path, {member + 'x'})

    def test_archive_member_hash_and_index_completeness(self):
        raw = fixtures.elf64()
        archive = parser.parse_archive(fixtures.archive(raw))
        path = 'CMakeFiles/unit.dir/unit.o'
        products = {path: {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}}
        result = r.index_members(archive, [path], products)
        self.assertEqual(result['index'], [('foo', 'unit.o')])
        bad = json.loads(json.dumps(archive))
        bad['symbol_index'] = []
        with self.assertRaisesRegex(r.Hold, 'index order/completeness'):
            r.index_members(bad, [path], products)
        bad = dict(products)
        bad[path] = {**products[path], 'sha256': '0' * 64}
        with self.assertRaisesRegex(r.Hold, 'raw bytes'):
            r.index_members(archive, [path], bad)

    def test_elf_index_normalization_and_exact_delta(self):
        raw = fixtures.elf64()
        inventory = parser.parse_elf(raw)
        canonical = r.canonical_elf(inventory, raw, [])
        self.assertEqual(r.delta_rows('unit.o', canonical, canonical,
                                      {'source_changed': []}), [])
        other = copy.deepcopy(canonical)
        other['sections'][4]['size'] += 1
        rows = r.delta_rows('unit.o', canonical, other, {'source_changed': ['unit.c']})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['field'], 'sections')
        self.assertEqual(rows[0]['object'], 'unit.o')
        affected = {'unit.o': {'source_changed': ['unit.c']}}
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'unit.c').write_bytes(b'old\nchanged\nlast\n')
            changed = {'unit.c': {2}}
            source_hashes = {'unit.c': r.sha_file(Path(tmp) / 'unit.c')}
            digest = r.sha_bytes(json.dumps(rows[0], sort_keys=True).encode())
            exception = {'delta_sha256': digest, 'object': 'unit.o',
                         'field': rows[0]['field'], 'identity': rows[0]['identity'],
                         'f': copy.deepcopy(rows[0]['f']), 'g': copy.deepcopy(rows[0]['g']),
                         'source_line': 'unit.c:2',
                         'source_line_sha256': r.sha_bytes(b'changed\n'),
                         'reason': 'Exact reviewed change'}
            self.assertEqual(r.apply_exceptions(rows, [exception], affected, changed,
                                                tmp, source_hashes), [])
            for bad_line in ('unit.c:1', 'unit.c:999999999'):
                with self.assertRaises(r.Hold):
                    r.apply_exceptions(rows, [{**exception, 'source_line': bad_line}],
                                       affected, changed, tmp, source_hashes)
            with self.assertRaisesRegex(r.Hold, 'content drift'):
                r.apply_exceptions(rows, [{**exception, 'source_line_sha256': '0' * 64}],
                                   affected, changed, tmp, source_hashes)
            with self.assertRaises(r.Hold):
                r.apply_exceptions(rows, [exception],
                                   {'unit.o': {'source_changed': []}}, changed, tmp, source_hashes)
            exception['g']['size'] += 1
            with self.assertRaisesRegex(r.Hold, 'does not match'):
                r.apply_exceptions(rows, [exception], affected, changed, tmp, source_hashes)
            (Path(tmp) / 'unit.c').write_bytes(b'old\nchanged\ntampered\n')
            with self.assertRaisesRegex(r.Hold, 'source file content drift'):
                r.apply_exceptions(rows, [{**exception, 'g': rows[0]['g']}],
                                   affected, changed, tmp, source_hashes)

    def test_changed_line_map_and_archive_header_hold(self):
        diff = (b'diff --git a/unit.c b/unit.c\n@@ -1,0 +2,1 @@\n+changed\n'
                b'diff --git a/lib.c b/lib.c\n@@ -3,1 +3,0 @@\n-old\n')
        self.assertEqual(r.changed_lines(diff, {'unit.c', 'lib.c'}),
                         {'unit.c': {2}, 'lib.c': set()})
        with self.assertRaises(r.Hold):
            r.changed_lines(diff.replace(b'b/unit.c', b'b/other.c'), {'unit.c', 'lib.c'})
        self.assertEqual(r.result_status([], []), 'PASS')
        self.assertEqual(r.result_status([], [{'changed_fields': ['raw_header_hex']}]), 'HOLD')
        self.assertEqual(r.result_status([], [{'changed_fields': ['body_sha256']}]), 'HOLD')

    def test_debug_string_whole_path_only(self):
        root = '/synthetic/source'
        raw = (root + '/unit.c').encode() + b'\0'
        section = {'name': '.debug_str', 'offset': 0, 'size': len(raw),
                   'raw_sha256': hashlib.sha256(raw).hexdigest()}
        self.assertEqual(r.debug_bytes(raw, section, [(root, '<SOURCE>')]),
                         hashlib.sha256(b'<SOURCE>/unit.c\0').hexdigest())
        with self.assertRaisesRegex(r.Hold, 'embedded or adjacent'):
            r.debug_bytes(b'x' + raw, {**section, 'size': len(raw) + 1},
                          [(root, '<SOURCE>')])

    def test_child_parser_synthetic_success_and_malformed_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_limits, old_imports, old_check_limits = r.child_limits, r.check_imports, r.check_limits
            r.child_limits = lambda: None
            r.check_imports = lambda: None
            r.check_limits = lambda: None
            try:
                raw = fixtures.elf64()
                (root / 'unit.o').write_bytes(raw)
                product = {'side': 'fb', 'path': 'unit.o',
                           'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    self.parse_synthetic('object', root / 'unit.o', root / 'receipt.json', product)
                parsed = json.loads(stdout.getvalue())
                self.assertEqual(parsed['sha256'], hashlib.sha256(raw).hexdigest())
                self.assertGreater(json.loads((root / 'receipt.json').read_text())['peak_rss_kib'], 0)
                (root / 'bad.o').write_bytes(b'not ELF')
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(Exception):
                    self.parse_synthetic('object', root / 'bad.o', root / 'bad.receipt.json',
                                  {'side': 'fb', 'path': 'bad.o',
                                   'sha256': hashlib.sha256(b'not ELF').hexdigest(), 'bytes': 7})
                self.assertFalse((root / 'bad.receipt.json').exists())
            finally:
                r.child_limits, r.check_imports, r.check_limits = old_limits, old_imports, old_check_limits

    def test_child_completion_samples_after_flush_and_output_failure_holds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = fixtures.elf64()
            (root / 'unit.o').write_bytes(raw)
            product = {'side': 'fb', 'path': 'unit.o',
                       'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
            state = {'peak': 10, 'imports': 0, 'limits': 0}
            class Usage:
                @property
                def ru_maxrss(self):
                    return state['peak']
            class DelayedPeak(io.StringIO):
                def flush(self):
                    state['peak'] = 20
                    super().flush()
            def imports():
                state['imports'] += 1
            def limits():
                state['limits'] += 1
            with mock.patch.object(r, 'child_limits'), mock.patch.object(
                    r, 'check_imports', side_effect=imports), mock.patch.object(
                    r, 'check_limits', side_effect=limits), mock.patch.object(
                    r.resource, 'getrusage', return_value=Usage()):
                with contextlib.redirect_stdout(DelayedPeak()):
                    self.parse_synthetic('object', root / 'unit.o', root / 'complete.json', product)
            receipt = json.loads((root / 'complete.json').read_text())
            self.assertEqual(receipt['peak_rss_kib'], 20)
            self.assertEqual(state['imports'], 2)
            self.assertEqual(state['limits'], 2)
            class BrokenOutput(io.StringIO):
                def write(self, text):
                    raise OSError('synthetic output failure')
            with mock.patch.object(r, 'child_limits'), mock.patch.object(
                    r, 'check_imports'), mock.patch.object(r, 'check_limits'):
                with contextlib.redirect_stdout(BrokenOutput()), self.assertRaises(OSError):
                    self.parse_synthetic('object', root / 'unit.o', root / 'failed.json', product)
            self.assertFalse((root / 'failed.json').exists())
            class BrokenFlush(io.StringIO):
                def flush(self):
                    raise OSError('synthetic flush failure')
            with mock.patch.object(r, 'child_limits'), mock.patch.object(
                    r, 'check_imports'), mock.patch.object(r, 'check_limits'):
                with contextlib.redirect_stdout(BrokenFlush()), self.assertRaises(OSError):
                    self.parse_synthetic('object', root / 'unit.o', root / 'flush-failed.json', product)
            self.assertFalse((root / 'flush-failed.json').exists())
            calls = {'imports': 0}
            def late_import_drift():
                calls['imports'] += 1
                if calls['imports'] == 2:
                    raise r.Hold('synthetic post-flush import drift')
            with mock.patch.object(r, 'child_limits'), mock.patch.object(
                    r, 'check_imports', side_effect=late_import_drift), mock.patch.object(
                    r, 'check_limits'):
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(r.Hold):
                    self.parse_synthetic('object', root / 'unit.o', root / 'post-flush-failed.json', product)
            self.assertFalse((root / 'post-flush-failed.json').exists())

    def test_parent_completion_samples_after_result_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = {'peak': 10}
            class Usage:
                @property
                def ru_maxrss(self):
                    return state['peak']
            original_dumps = json.dumps
            def delayed_dumps(value, *args, **kwargs):
                if value.get('status') == 'PASS':
                    state['peak'] = 20
                return original_dumps(value, *args, **kwargs)
            def final_imports():
                self.assertTrue((root / 'RESULT.json').is_file())
            with mock.patch.object(r.json, 'dumps', side_effect=delayed_dumps), mock.patch.object(
                    r, 'check_imports', side_effect=final_imports), mock.patch.object(
                    r.resource, 'getrlimit', return_value=(r.AS_BYTES, r.AS_BYTES)), mock.patch.object(
                    r.resource, 'getrusage', return_value=Usage()):
                r.write_result_and_parent_completion(root, {'status': 'PASS'})
            completion = json.loads((root / 'PARENT-COMPLETED.json').read_text())
            self.assertEqual(completion['parent_peak_rss_kib'], 20)
            self.assertEqual(completion['result_sha256'], r.sha_file(root / 'RESULT.json'))
            second = root / 'failed-output'
            second.mkdir()
            with self.assertRaises(TypeError):
                r.write_result_and_parent_completion(second, {'status': 'PASS', 'bad': object()})
            self.assertFalse((second / 'PARENT-COMPLETED.json').exists())
            third = root / 'late-limit-drift'
            third.mkdir()
            with mock.patch.object(r, 'check_imports'), mock.patch.object(
                    r.resource, 'getrlimit', return_value=(r.AS_BYTES - 1, r.AS_BYTES)):
                with self.assertRaisesRegex(r.Hold, 'after result'):
                    r.write_result_and_parent_completion(third, {'status': 'PASS'})
            self.assertTrue((third / 'RESULT.json').is_file())
            self.assertFalse((third / 'PARENT-COMPLETED.json').exists())

    def test_seal_parser_rejects_duplicate_and_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'SHA256SUMS'
            path.write_text('0' * 64 + '  x\n' + '1' * 64 + '  x\n')
            with self.assertRaisesRegex(r.Hold, 'duplicate'):
                r.seal_entries(path)
            path.write_text('0' * 64 + '  ../x\n')
            with self.assertRaisesRegex(r.Hold, 'unsafe'):
                r.seal_entries(path)

    def test_review_roles_and_exception_digest_are_pinned(self):
        with mock.patch.object(r, 'freeze_import_resolution'):
            with self.assertRaisesRegex(r.Hold, 'binding digest absent'):
                r.review_gate(Path('/nonexistent/synthetic-release.json'))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binding = {'runner_manifest_sha256': 'a' * 64,
                       'release_sha256': 'd' * 64,
                       'exception_manifest_sha256': 'c' * 64,
                       'exception_file_sha256': 'b' * 64}
            release = {'runner_manifest_sha256': binding['runner_manifest_sha256'],
                       'exception_file_sha256': binding['exception_file_sha256']}
            authority = {}
            for role in ('independent', 'root'):
                folder = root / role
                folder.mkdir()
                decision = folder / 'RELEASE.json'
                decision.write_text(json.dumps({'role': role, 'verdict': 'RELEASE',
                    **{key: binding[key] for key in (
                        'runner_manifest_sha256', 'release_sha256',
                        'exception_manifest_sha256', 'exception_file_sha256')}}))
                seal = folder / 'SHA256SUMS-REVIEW'
                seal.write_text(f'{r.sha_file(decision)}  RELEASE.json\n')
                authority[role] = seal
                binding[f'{role}_review_seal_sha256'] = r.sha_file(seal)
            with mock.patch.object(r, 'REVIEW_AUTHORITY', authority):
                r.trusted_review_decisions(release, binding)
                forged = root / 'self-authored.json'
                forged.write_text(json.dumps({'role': 'root', 'verdict': 'RELEASE', **binding}))
                authority['root'] = forged
                with self.assertRaises(r.Hold):
                    r.trusted_review_decisions(release, binding)
                authority['root'] = authority['independent']
                with self.assertRaises(r.Hold):
                    r.trusted_review_decisions(release, binding)
                authority['root'] = root / 'root/SHA256SUMS-REVIEW'
                binding['exception_file_sha256'] = 'e' * 64
                with self.assertRaisesRegex(r.Hold, 'review did not release exact'):
                    r.trusted_review_decisions(release, binding)

    def test_child_signal_and_malformed_stdout_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'build').mkdir()
            output = root / 'output'
            output.mkdir()
            raw = fixtures.elf64()
            (root / 'build/unit.o').write_bytes(raw)
            product = {'path': 'unit.o', 'kind': 'object', 'bytes': len(raw),
                       'sha256': hashlib.sha256(raw).hexdigest()}
            old_run, old_call = r.RUN['fb'], r.subprocess.run
            r.RUN['fb'] = root
            class Exit:
                returncode = -9
            launches = []
            def synthetic_child(*args, **kwargs):
                launches.append((args, kwargs))
                return Exit()
            r.subprocess.run = synthetic_child
            try:
                with self.assertRaisesRegex(r.Hold, 'child failed'):
                    r.run_child('fb', product, output, root / 'release.json', 'a' * 64)
                self.assertEqual(launches[0][0][0][-1], '3')
                self.assertEqual(launches[0][1]['pass_fds'], (3,))
                child = json.loads((output / 'fb-unit.o.child.json').read_text())
                self.assertEqual(child['exit_code'], -9)
                self.assertEqual(child['signal'], 9)
                self.assertEqual(child['status'], 'SIGNAL')
            finally:
                r.RUN['fb'], r.subprocess.run = old_run, old_call

    def test_timeout_writes_bounded_failure_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'build').mkdir()
            output = root / 'output'
            output.mkdir()
            raw = fixtures.elf64()
            (root / 'build/unit.o').write_bytes(raw)
            product = {'path': 'unit.o', 'kind': 'object', 'bytes': len(raw),
                       'sha256': hashlib.sha256(raw).hexdigest()}
            with mock.patch.dict(r.RUN, {'fb': root}), mock.patch.object(
                    r.subprocess, 'run', side_effect=subprocess.TimeoutExpired('synthetic', 1)):
                with self.assertRaisesRegex(r.Hold, 'child failed'):
                    r.run_child('fb', product, output, root / 'release.json', 'a' * 64)
            child = json.loads((output / 'fb-unit.o.child.json').read_text())
            self.assertEqual(child['status'], 'TIMEOUT')
            self.assertIsNone(child['exit_code'])
            self.assertEqual(child['stdout_bytes'], 0)
            self.assertIn('limits', child)
            second = root / 'spawn-output'
            second.mkdir()
            with mock.patch.dict(r.RUN, {'fb': root}), mock.patch.object(
                    r.subprocess, 'run', side_effect=subprocess.SubprocessError('synthetic')):
                with self.assertRaisesRegex(r.Hold, 'child failed'):
                    r.run_child('fb', product, second, root / 'release.json', 'a' * 64)
            spawned = json.loads((second / 'fb-unit.o.child.json').read_text())
            self.assertEqual(spawned['status'], 'SPAWN_ERROR')
            self.assertEqual(spawned['error_type'], 'SubprocessError')

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'build').mkdir()
            output = root / 'output'
            output.mkdir()
            raw = fixtures.elf64()
            (root / 'build/unit.o').write_bytes(raw)
            old_run, old_call = r.RUN['fb'], r.subprocess.run
            r.RUN['fb'] = root
            class Exit:
                returncode = 0
            def malformed(*args, **kwargs):
                kwargs['stdout'].write(b'{bad json')
                receipt = Path(args[0][9])
                receipt.write_text(json.dumps({'input_sha256': hashlib.sha256(raw).hexdigest(),
                    'side': 'fb', 'path': 'unit.o', 'bytes': len(raw),
                    'kind': 'object', 'parser_sha256':
                    '4944bc67066bf55c26e9da4a3fa075458b4c5f783164a88e0730d5ed9f6f23a4',
                    'peak_rss_kib': 1}))
                return Exit()
            r.subprocess.run = malformed
            try:
                with self.assertRaises(json.JSONDecodeError):
                    r.run_child('fb', product, output, root / 'release.json', 'a' * 64)
                child = json.loads((output / 'fb-unit.o.child.json').read_text())
                self.assertEqual(child['status'], 'INVALID_OUTPUT')
            finally:
                r.RUN['fb'], r.subprocess.run = old_run, old_call


if __name__ == '__main__':
    unittest.main()
