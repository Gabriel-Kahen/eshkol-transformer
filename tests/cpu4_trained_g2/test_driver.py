"""Pure resources/overlay and mocked subprocess custody; no native execution."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from tests.d2.test_resources import load_d1_resource
from tests.cpu4_resume import normal_driver as plain
from . import normal_driver as driver
from .check_output import check, MODES
from .test_checker import records,text


class Driver(unittest.TestCase):
    def test37_unique_TUs_only_authorized_readonly_overlay(self):
        roster=driver.load_roster();before=copy.deepcopy(roster)
        overlay=json.loads(driver.OVERLAY.read_text());units=driver.plan(roster,overlay)
        self.assertEqual(roster,before);self.assertEqual(len(units),37)
        self.assertEqual(len({u['source'] for u in units}),37)
        for old,new in zip(roster['translation_units'],units):
            self.assertEqual(new,dict(source=old['source'],defines=old['defines']+driver.TEST_FLAGS.get(old['source'],[])))
        self.assertEqual(units[-2:],driver.OBSERVERS)
        self.assertEqual(driver.TEST_FLAGS,{'native/o2_optimizer.c':['ET_O2_TESTING'],
            'src/eshkol_transformer/m3_call_f32_integration.c':['ET_F32_TENSOR_TESTING']})
        for field in ('extra','duplicate','flag','san'):
            bad=copy.deepcopy(overlay)
            if field=='extra':bad['additional_translation_units'].append({'source':'native/extra.c','defines':[]})
            if field=='duplicate':bad['additional_translation_units'].append(bad['additional_translation_units'][0])
            if field=='flag':bad['per_tu_defines']['native/checkpoint_io.c']=['ET_CHECKPOINT_IO_TESTING']
            if field=='san':bad['mode']='sanitize'
            with self.assertRaises(ValueError):driver.plan(roster,bad)
        driver.strict_dependencies(driver.ROOT)
        self.assertEqual(driver.preflight(driver.ROOT, roster, overlay), units)
        # The composed source passes; either original native leaf must still
        # fail the real repair barrier before native compilation.
        read_bytes = Path.read_bytes
        for relative in driver.REQUIRED_REPAIR:
            stale_path = driver.ROOT / relative
            baseline = subprocess.check_output(
                ['git', 'show', 'fd0db7270ca0212f73fc64337379d1ada32d0154:' + relative],
                cwd=driver.ROOT)
            def stale_read(path, target=stale_path, data=baseline):
                return data if path == target else read_bytes(path)
            with patch.object(Path, 'read_bytes', stale_read):
                with self.assertRaisesRegex(ValueError, 'O2 repair composition'):
                    driver.preflight(driver.ROOT, roster, overlay)

    def test_immutable_resource_bytes_same_as_plain(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'plain';b=Path(tmp)/'g2';a.mkdir();b.mkdir()
            plain.write_resources(a);driver.write_resources(b)
            for p in sorted((a/'resources').rglob('*')):
                if p.is_file():self.assertEqual(p.read_bytes(),(b/p.relative_to(a)).read_bytes())
            for mode in MODES:
                for label,expected in [('train',(3,197,3,41,7)),('heldout',(197,197,3,41,7))]:
                    data,_=load_d1_resource(b/'resources'/label,expected_fingerprint=driver.FINGERPRINT,expected_vocab=256)
                    self.assertEqual(data,(expected,))

    def mocked_gate(self,out,fail_run=None,repeat_mismatch=False,checkpoint_mismatch=False,custody_failure=False):
        commands=[];runs=[]
        def run(argv,**kwargs):
            commands.append(argv);stdout=kwargs['stdout']
            if argv[:2]==['git','rev-parse']:stdout.write(b'commit\ntree\n')
            elif '-c' in argv:
                target=Path(argv[argv.index('-o')+1]);target.write_bytes(b'o')
                Path(argv[argv.index('-MF')+1]).write_text('out.o: tests/cpu4_trained_g2/generation.esk\n')
            elif '--emit-object' in argv:
                target=Path(argv[argv.index('-o')+1])
                for suffix in ('','.ll','.bc'):Path(str(target)+suffix).write_bytes(b'o')
                Path(argv[argv.index('--emit-depfile')+1]).write_text('out.o: tests/cpu4_trained_g2/runtime.esk\n')
            elif '-o' in argv:Path(argv[argv.index('-o')+1]).write_bytes(b'ELF')
            elif argv[:2]==['ar','rcsD']:Path(argv[2]).write_bytes(b'archive')
            elif '900s' in argv:
                mode=argv[-1];checkpoint=Path(argv[-2]);runs.append((mode,checkpoint.name))
                if len(runs)==fail_run:return subprocess.CompletedProcess(argv,37)
                data=text(records(mode)).encode()
                if repeat_mismatch and checkpoint.name=='two-token.saved2.c2':data=data.replace(b'RNG 1 41',b'RNG 1 42')
                stdout.write(data);checkpoint.write_bytes(b'C2'+(b'x' if checkpoint_mismatch and checkpoint.name=='first-eos.saved1.c2' else b''))
            elif argv[0]==driver.sys.executable:
                check(Path(argv[-2]).read_text(),argv[-1]);stdout.write(b'CHECK-PASS\n')
            elif argv[0]=='cmp' and Path(argv[1]).read_bytes()!=Path(argv[2]).read_bytes():
                return subprocess.CompletedProcess(argv,1)
            return subprocess.CompletedProcess(argv,0)
        with patch.object(driver,'preflight',return_value=[{'source':'native/o2_optimizer.c','defines':[]}]),patch.object(driver.subprocess,'run',side_effect=run):
            if custody_failure:
                with patch.object(driver,'inventory_depfiles',side_effect=ValueError('custody')):
                    status=driver.gate(out,Path('/runner'),Path('/cc'),Path('/cxx'),Path('/compiler'))
            else:status=driver.gate(out,Path('/runner'),Path('/cc'),Path('/cxx'),Path('/compiler'))
        return status,commands,runs

    def test_four_actual_process_slots_strict_checks_and_sealed_discovery_mocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'fresh';status,commands,runs=self.mocked_gate(out)
            self.assertEqual(status,0)
            self.assertEqual(runs,[(m,f'{m}.saved{r}.c2') for m in MODES for r in (1,2)])
            self.assertEqual(sum(a[0]==driver.sys.executable for a in commands),4)
            self.assertEqual(sum(a[0]=='cmp' for a in commands),5)
            for row in (out/'SHA256SUMS').read_text().splitlines():
                digest,path=row.split('  ');self.assertEqual(hashlib.sha256((out/path).read_bytes()).hexdigest(),digest)
            self.assertEqual((out/'driver-custody.exit').read_text(),'0\n')

    def test_failure_stops_first_and_preserves_exit_custody_and_mode_parity(self):
        for failure,expected in [('run',37),('custody',74),('repeat',1),('checkpoint',1)]:
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as tmp:
                status,commands,runs=self.mocked_gate(Path(tmp)/'fresh',fail_run=1 if failure in ('run','custody') else None,
                    custody_failure=failure=='custody',repeat_mismatch=failure=='repeat',checkpoint_mismatch=failure=='checkpoint')
                self.assertEqual(status,expected)
                if failure in ('run','custody'):self.assertEqual(len(runs),1)


if __name__=='__main__':unittest.main()
