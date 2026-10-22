"""Only source/resource/pure-Python/mocked driver checks. No compiler/process run."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from tests.cpu4_trainer_control.test_source_contract import parse, walk
from tests.d2.test_resources import load_d1_resource
from . import normal_driver as driver
from .check_resume import check as check_resume


def bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def synthetic_stdout():
    tail = f'{bits(.5)} 1082130432 {bits(2.)} {bits(.5)} 4 1 1000 250 249'
    return ('\n'.join([
        f'CPU4-HELDOUT-OBS baseline {bits(5.)} 1082130432 {bits(150.)} {bits(.25)} 4 1 0 0 0',
        'CPU4-RESUME-STATE saved125 500 125 124 4768 9536 125 1065353216',
        'CPU4-HELDOUT-OBS post ' + tail,
        'CPU4-RESUME-STATE restored125 500 125 124 4768 9536 125 1065353216',
        'CPU4-RESUME-STATE equal250 1000 250 249 4768 9536 250 1065353216',
        'CPU4-RESUME-HELDOUT ' + tail, 'CPU4-RESUME-PASS 770 250 125 375 1192 2384 0']) + '\n').encode()


class Driver(unittest.TestCase):
    def test_frontend_exact_actual_load_resolution_and_successful_cleanup(self):
        forms = parse(driver.CALLER.read_text())
        self.assertEqual(forms[:3], [['load', 'cpu4_v1_private_root.esk'],
            ['load', '../cpu4_heldout/body.esk'], ['load', 'body.esk']])
        roster = driver.load_roster()
        for _, name in forms[:3]:
            candidates = [driver.CALLER.parent / name] + [driver.ROOT / d / name for d in roster['source_include_dirs']]
            self.assertTrue(any(path.is_file() for path in candidates), name)
        self.assertEqual(len(forms), 6)
        functions = {f[1][0]: f for f in forms[3:5]}
        runtime = functions['cpu4-resume-driver-runtime!']
        heads = [n[0] for n in walk(runtime) if n and isinstance(n[0], str)]
        self.assertEqual(heads.count('cpu4-resume-driver-open'), 3)
        self.assertEqual(heads.count('cpu4-heldout-resume-runtime!'), 1)
        self.assertEqual(heads.count('d2-token-dataset-close!'), 3)
        self.assertLess(heads.index('cpu4-heldout-resume-runtime!'), heads.index('d2-token-dataset-close!'))
        for name in ('t2-wave2-tokenizer-byte', 'k2-public-capability-discover', 'c2-persistence-policy-internal'):
            self.assertEqual(heads.count(name), 1)
        self.assertIn('\\"run.seed\\":1729,\\"training.accumulation-steps\\":1', driver.CALLER.read_text())
        self.assertNotIn('guard', heads)
        guard = forms[-1]
        self.assertEqual(guard[0], 'guard')
        self.assertEqual(guard[2:], [['cpu4-resume-driver-runtime!']])
        handler = guard[1][1]
        self.assertEqual(handler[-1], ['exit', '1'])
        for name in ('transformer-error-category', 'transformer-error-operation',
                     'transformer-error-message', 'transformer-error-details'):
            self.assertEqual(sum(n and n[0] == name for n in walk(handler)), 1)
        self.assertIn(['write', 'caught'], list(walk(handler)))
        self.assertIn(['display', 'cpu4-resume-checks'], list(walk(handler)))
        self.assertIn(['display', 'cpu4-resume-steps'], list(walk(handler)))

    def test_minimal_actual_import_overlay_and_production_immutability(self):
        roster = driver.load_roster()
        before = copy.deepcopy(roster)
        overlay = json.loads(driver.OVERLAY.read_text())
        selected = driver.plan(roster, overlay)
        self.assertEqual(roster, before)
        self.assertEqual(selected[-1], driver.OBSERVER)
        self.assertEqual(len(selected), len(roster['translation_units'])+1)
        for old, new in zip(roster['translation_units'], selected):
            self.assertEqual(new, dict(source=old['source'], defines=old['defines']+driver.TEST_FLAGS.get(old['source'], [])))
        observer = (driver.ROOT / driver.OBSERVER['source']).read_text()
        self.assertIn('et_o2_test_optimizer_moment_bits_v1', observer)
        self.assertNotIn('ET_F32_TENSOR_TESTING', observer)
        for mutate in ('san', 'owner', 'hook'):
            bad = copy.deepcopy(overlay)
            if mutate == 'san': bad['mode'] = 'sanitize'
            if mutate == 'owner': bad['additional_translation_units'].append({'source': 'native/o2_optimizer.c', 'defines': []})
            if mutate == 'hook': bad['per_tu_defines']['native/checkpoint_io.c'] = ['ET_CHECKPOINT_IO_TESTING']
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                driver.plan(roster, bad)

    def test_existing_caller_pins_fail_closed_before_common_plan(self):
        self.assertIn("if __name__ == '__main__':", Path(driver.__file__).read_text())
        self.assertIn('len(sys.argv) != 6', Path(driver.__file__).read_text())
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            directory = repo / 'tests/cpu4_resume'
            directory.mkdir(parents=True)
            rows = []
            for index in range(8):
                path = repo / f'input{index}'
                path.write_text(str(index))
                rows.append(hashlib.sha256(path.read_bytes()).hexdigest() + '  ' + path.name)
            (directory / 'DEPENDENCIES.sha256').write_text('\n'.join(rows)+'\n')
            driver.strict_dependencies(repo)
            (repo / 'input7').write_text('changed')
            with patch.object(driver, 'plan') as plan, self.assertRaisesRegex(ValueError, 'stale'):
                driver.preflight(repo, {}, {})
            plan.assert_not_called()
        with patch.object(driver, 'strict_dependencies'), patch.object(driver, 'plan'), \
             patch.object(driver, 'manifest', return_value='stale'):
            with self.assertRaisesRegex(ValueError, 'root-owned'):
                driver.preflight(driver.ROOT, {}, {})

    def test_exact_resource_bytes_and_distinct_first_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            train, heldout = driver.write_resources(out)
            for path, expected in ((train, (3,197,3,41,7)), (heldout, (197,197,3,41,7))):
                shards, _ = load_d1_resource(path, expected_fingerprint=driver.FINGERPRINT, expected_vocab=256)
                self.assertEqual(shards, (expected,))
            self.assertNotEqual((train/'manifest.etm').read_bytes(), (heldout/'manifest.etm').read_bytes())
            frozen = {
                train/'manifest.etm': ('a11ef524934c63d8a16df70bc1df3690f16eb982d9b4c00c840cba6020f5ca5d', 280),
                train/'shard-0000000000000000.ets': ('174b0b7101410295b11d2647b4a7fd5c34882cbfd1ab1c818530fffd8183e5e2', 248),
                heldout/'manifest.etm': ('14e8dbcf0a79f3f98462e1d49f7569ff7a0f29aeb8c2d7f1e2c3165b6b1d1753', 280),
                heldout/'shard-0000000000000000.ets': ('2338d78d794ccb1ace21bec27004d11e05cca63ab05f3489578f81eb3b32e13b', 248),
            }
            for path, (digest, size) in frozen.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
                self.assertEqual(path.stat().st_size, size)
            for line in (out/'resource-bytes.sha256').read_text().splitlines():
                digest, path = line.split('  ')
                self.assertEqual(hashlib.sha256((out/'resources'/path).read_bytes()).hexdigest(), digest)

    def mock_gate(self, out, native_status=0, custody_failure=False, repeat_mismatch=False, checkpoint_mismatch=False):
        commands = []
        units = [{'source': 'native/o2_optimizer.c', 'defines': ['ET_O2_TESTING']}]
        def run(argv, **kwargs):
            commands.append(argv)
            stdout = kwargs['stdout']
            if argv[:2] == ['git', 'rev-parse']:
                stdout.write(b'commit\ntree\n')
            elif argv[0] == 'git':
                pass
            elif '-c' in argv:
                if native_status: return subprocess.CompletedProcess(argv, native_status)
                target = Path(argv[argv.index('-o')+1]); target.write_bytes(b'object')
                Path(argv[argv.index('-MF')+1]).write_text('out.o: native/o2_optimizer.c\n')
            elif '--emit-object' in argv:
                target = Path(argv[argv.index('-o')+1])
                for suffix in ('', '.ll', '.bc'): Path(str(target)+suffix).write_bytes(b'compiled')
                Path(argv[argv.index('--emit-depfile')+1]).write_text('out.o: tests/cpu4_resume/runtime.esk\n')
            elif '-o' in argv:
                Path(argv[argv.index('-o')+1]).write_bytes(b'ELF')
            elif argv[0] == 'ar' and argv[1] == 'rcsD':
                Path(argv[2]).write_bytes(b'archive')
            elif '900s' in argv:
                checkpoint = Path(argv[-1])
                data = synthetic_stdout()
                if repeat_mismatch and checkpoint.name == 'saved2.c2':
                    data = data.replace(str(bits(.5)).encode(), str(bits(.25)).encode())
                stdout.write(data)
                checkpoint.write_bytes(b'C2'+(b'different' if checkpoint_mismatch and checkpoint.name=='saved2.c2' else b''))
            elif argv[0] == sys_executable:
                check_resume(Path(argv[-1]).read_text())
                stdout.write(b'CHECK-PASS\n')
                self.assertEqual(kwargs['env']['PYTHONDONTWRITEBYTECODE'], '1')
            elif argv[0] == 'cmp' and Path(argv[1]).read_bytes() != Path(argv[2]).read_bytes():
                return subprocess.CompletedProcess(argv, 1)
            return subprocess.CompletedProcess(argv, 0)
        sys_executable = driver.sys.executable
        with patch.object(driver, 'preflight', return_value=units), patch.object(driver.subprocess, 'run', side_effect=run):
            if custody_failure:
                with patch.object(driver, 'inventory_depfiles', side_effect=ValueError('custody failure')):
                    result = driver.gate(out, Path('/runner'), Path('/cc'), Path('/cxx'), Path('/source'))
            else:
                result = driver.gate(out, Path('/runner'), Path('/cc'), Path('/cxx'), Path('/source'))
        return result, commands

    def test_mocked_success_runs_real_checker_twice_and_seals_discovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)/'fresh'
            status, commands = self.mock_gate(out)
            self.assertEqual(status, 0)
            self.assertEqual(sum('900s' in argv for argv in commands), 2)
            self.assertEqual(sum(argv[0] == driver.sys.executable for argv in commands), 2)
            self.assertEqual((out/'driver-program.exit').read_text(), '0\n')
            self.assertEqual((out/'driver-custody.exit').read_text(), '0\n')
            self.assertTrue(json.loads((out/'observed-dependencies.json').read_text()))
            for line in (out/'SHA256SUMS').read_text().splitlines():
                digest, name = line.split('  ', 1)
                self.assertEqual(hashlib.sha256((out/name).read_bytes()).hexdigest(), digest)
            self.assertEqual((out/'SHA256SUMS.sha256').read_text().split()[0], hashlib.sha256((out/'SHA256SUMS').read_bytes()).hexdigest())
            with self.assertRaises(ValueError):
                driver.gate(out, Path('/runner'), Path('/cc'), Path('/cxx'), Path('/source'))

    def test_raw_failure_custody_failure_and_repeat_negatives(self):
        for case in ('native', 'custody', 'stdout', 'checkpoint'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                out = Path(temporary)/'fresh'
                status, _ = self.mock_gate(out, native_status=37 if case in ('native','custody') else 0,
                    custody_failure=case=='custody', repeat_mismatch=case=='stdout', checkpoint_mismatch=case=='checkpoint')
                self.assertEqual(status, 74 if case=='custody' else 37 if case=='native' else 1)
                self.assertEqual((out/'driver-program.exit').read_text(), '37\n' if case in ('native','custody') else '1\n')
                self.assertEqual((out/'driver-custody.exit').read_text(), '74\n' if case=='custody' else '0\n')

    def test_actual_discovery_parser_handles_escaped_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            depfile = Path(temporary)/'actual.d'
            depfile.write_text('out.o: native/a.c \\\n native/a.h path\\ with\\ spaces/header.h\n')
            self.assertEqual(driver.dependencies(depfile), [(driver.ROOT/name).resolve() for name in
                ('native/a.c', 'native/a.h', 'path with spaces/header.h')])


if __name__ == '__main__':
    unittest.main()
