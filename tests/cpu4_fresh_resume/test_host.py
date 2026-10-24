"""Host/source tests. Synthetic C2 bytes and fake transcripts are NOT training evidence."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from tests.cpu4_fresh_resume import check, driver
from tests.cpu4_trainer_control.test_source_contract import parse, walk


def fixture(updates):
    """Deliberately non-numeric structural parser double, generated in memory."""
    model = bytearray(10973)
    model[:16] = bytes.fromhex('894553484b4f4c434b50540d0a1a0a00')
    struct.pack_into('<QQ', model, 64, 2077, 8864)
    struct.pack_into('<II', model, 80, 15, 1)
    model[-32:] = hashlib.sha256(check.C1_DOMAIN+model[:-32]).digest()
    data = bytearray(25504)
    data[:16] = bytes.fromhex('894553484b54524e53544154450d0a00')
    struct.pack_into('<HHIIIQ', data, 16, 1, 0, 256, 1, 1, 0)
    for offset, value in {40:25504, 48:256, 56:4707, 64:4963, 72:20509,
                          96:10973, 104:2656, 112:9536, 120:1107}.items():
        struct.pack_into('<Q', data, offset, value)
    struct.pack_into('<IIII', data, 80, 15, 14, 28, 43)
    struct.pack_into('<7Q', data, 184, 4*updates, updates, updates-1, 1, 1729, 0, 0)
    data[4963:4963+len(model)] = model
    return reseal(data)


def reseal(data):
    data = bytearray(data)
    data[-32:] = hashlib.sha256(check.C2_DOMAIN+data[:-32]).digest()
    return bytes(data)


def transcript(mode):
    if mode == 'produce':
        rows = [check.words('CPU4-HELDOUT-OBS baseline', check.BASELINE), check.state('a125',125),
                check.words('CPU4-HELDOUT-OBS post',check.METRICS), check.state('a250',250),
                check.words('CPU4-FRESH-METRICS produce',check.METRICS), 'CPU4-FRESH-PASS produce 250 0']
    elif mode == 'consume':
        rows = [check.state('b125',125),'CPU4-FRESH-RESTORED 25504 1192 2384',check.state('b250',250),
                check.words('CPU4-FRESH-METRICS consume',check.METRICS),'CPU4-FRESH-PASS consume 125 0']
    else:
        error = 'invalid-argument cpu4-checkpoint-factory-create' if mode == 'reject-identity' else 'corrupt-data checkpoint-load'
        rows = [f'CPU4-FRESH-REJECT {mode} {error}',f'CPU4-FRESH-PASS {mode} 0 0']
    return '\n'.join(rows)+'\n'


class Host(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.external = self.root/'external.c2'
        self.external.write_bytes(fixture(125))
        self.sha = check.digest(self.external)
        self.pins = patch.object(check,'A125_SHA',self.sha)
        self.pins.start()
        self.addCleanup(self.pins.stop)

    def artifacts(self):
        out=self.root/'out';out.mkdir()
        for name,u in [('producer.a125.c2',125),('producer.a250.c2',250),
                       ('consumer.restored125.c2',125),('consumer.b250.c2',250)]:
            (out/name).write_bytes(fixture(u))
        return out

    def test_complete_artifacts_and_wrong_reference(self):
        out=self.artifacts()
        self.assertEqual(len(check.compare(out,self.external)),4)
        with patch.object(check,'A125_SHA','0'*64), self.assertRaises(ValueError):
            check.compare(out,self.external)

    def test_corruption_truncation_symlink_controls_and_shape(self):
        path=self.root/'bad.c2'
        mutations=[]
        data=bytearray(fixture(125));data[-1]^=1;mutations.append(data)
        mutations.append(fixture(125)[:-1])
        for offset in (80,112,184,216,240):
            data=bytearray(fixture(125));data[offset]^=1;mutations.append(reseal(data))
        for data in mutations:
            path.write_bytes(data)
            with self.assertRaises(ValueError):check.checkpoint(path,125)
        path.unlink();path.symlink_to(self.external)
        with self.assertRaises(ValueError):check.checkpoint(path,125)

    def test_rechecksummed_differences_across_all_serialized_sections_rejected(self):
        out=self.artifacts();path=out/'consumer.b250.c2'
        for offset in (300,4963+200,4963+10973,25000):
            data=bytearray(fixture(250));data[offset]^=1;path.write_bytes(reseal(data))
            with self.assertRaises(ValueError):check.compare(out,self.external)
        path.write_bytes(fixture(125))
        with self.assertRaises(ValueError):check.compare(out,self.external)

    def test_transcripts_fail_closed_on_missing_cleanup_steps_metrics_and_extra_lines(self):
        for mode in ('produce','consume','reject-corrupt','reject-identity'):
            good=transcript(mode);check.transcript(mode,good)
            for bad in (good+'extra\n',good[:-1],good.replace(' 0\n',' 1\n'),good.replace('PASS','FAIL')):
                with self.assertRaises(ValueError):check.transcript(mode,bad)
        with self.assertRaises(ValueError):check.transcript('unknown','')

    def fake(self,out,same_pid=False,bad_producer=False):
        class Fake:
            receipts={}
            def __call__(inner,label,argv,env,seconds):
                mode=argv[1];prefix=Path(argv[-1]);inner.receipts[label]={'pid':100 if same_pid else 100+len(inner.receipts)}
                (out/(label+'.stderr')).write_bytes(b'')
                if mode=='produce':
                    prefix.with_suffix('.a125.c2').write_bytes(fixture(250 if bad_producer else 125))
                    prefix.with_suffix('.a250.c2').write_bytes(fixture(250))
                elif mode=='consume':
                    prefix.with_suffix('.restored125.c2').write_bytes(fixture(125))
                    prefix.with_suffix('.b250.c2').write_bytes(fixture(250))
                return transcript(mode).encode()
        return Fake()

    def test_scenario_order_exact_inputs_and_negative_payloads(self):
        out=self.root/'out';out.mkdir();run=self.fake(out)
        driver.scenarios(out,'never-executed','train','heldout',self.external,{},run)
        self.assertEqual(list(run.receipts),['produce','consume','corrupt','truncated','identity'])
        self.assertEqual((out/'truncated.c2').read_bytes(),self.external.read_bytes()[:-1])
        self.assertNotEqual((out/'corrupt.c2').read_bytes(),self.external.read_bytes())
        self.assertEqual(check.digest(self.external),self.sha)
        self.assertEqual(json.loads((out/'oracle.json').read_text())['actual_training_steps']['consume'],125)

    def test_bad_producer_or_same_pid_prevents_adoption(self):
        for index,kwargs in enumerate(({'bad_producer':True},{'same_pid':True})):
            out=self.root/str(index);out.mkdir();run=self.fake(out,**kwargs)
            with self.assertRaises(ValueError):
                driver.scenarios(out,'never-executed','train','heldout',self.external,{},run)
            self.assertFalse((out/'oracle.json').exists())
            if index==0:self.assertNotIn('consume',run.receipts)

    def test_direct_python_host_pid_and_timeout_receipts(self):
        run=driver.Runner(self.root)
        run('host',[sys.executable,'-c','print("host-only")'])
        self.assertIsInstance(run.receipts['host']['pid'],int)
        with self.assertRaises(subprocess.CalledProcessError):
            run('timeout',[sys.executable,'-c','import time; time.sleep(10)'],seconds=.02)
        receipt=json.loads((self.root/'timeout.process.json').read_text())
        self.assertEqual(receipt['status'],124)
        self.assertTrue(receipt['timed_out'])
        self.assertLess(receipt['observed_returncode'],0)

    def test_input_immutability_checked_even_when_build_fails(self):
        class FakeRunner:
            def __init__(inner,out):pass
            def __call__(inner,label,argv,*args,**kwargs):
                return b'head\ntree\n' if label.endswith(('before','after')) else b''
        def failed_build(*args):
            self.external.write_bytes(fixture(250))
            raise ValueError('synthetic build failure')
        out=self.root/'gate'
        with patch.object(driver,'Runner',FakeRunner),patch.object(driver,'preflight',return_value=[]),patch.object(driver,'build',side_effect=failed_build):
            self.assertEqual(driver.gate(out,Path('never'),Path('never'),Path('never'),Path('never'),self.external),74)
        self.assertIn('external', (out/'custody-error.txt').read_text())
        self.assertFalse((out/'oracle.json').exists())


class Source(unittest.TestCase):
    def test_b0_pins_and_roster_remain_exact(self):
        self.assertEqual(len(driver.preflight(driver.ROOT,driver.load_roster(),json.loads(driver.OVERLAY.read_text()))),36)
        from tests.cpuv1_resource.projection import off, tokens
        for line in (driver.ROOT/driver.DEPENDENCIES).read_text().splitlines():
            digest,path=line.split('  ',1)
            original=subprocess.check_output(['git','show','b0af97d10a96fc62ef414a26953582f5c79ad642:'+path],cwd=driver.ROOT)
            if path == 'tests/cpu4_heldout/inspect.c':
                self.assertEqual(tokens(off((driver.ROOT/path).read_text())),tokens(original.decode()))
            else:
                self.assertEqual(hashlib.sha256(original).hexdigest(),digest)

    def test_single_factory_before_suffix_and_independent_uninterrupted_producer(self):
        text=(driver.ROOT/'tests/cpu4_fresh_resume/body.esk').read_text()
        forms=parse(text);self.assertTrue(all(f[0]=='define' for f in forms))
        functions={f[1][0]:f for f in forms}
        def calls(name,callee):return [f for f in walk(functions[name]) if f and f[0]==callee]
        self.assertEqual(len(calls('cpu4-fresh-consume!','cpu4-checkpoint-factory-create-internal!')),1)
        self.assertFalse(calls('cpu4-fresh-produce!','cpu4-checkpoint-factory-create-internal!'))
        self.assertEqual(calls('cpu4-fresh-produce!','cpu4-resume-run!'),[['cpu4-resume-run!','a','1','125'],['cpu4-resume-run!','a','126','250']])
        self.assertEqual(calls('cpu4-fresh-consume!','cpu4-resume-run!'),[['cpu4-resume-run!','b','126','250']])
        consumer=text[text.index('(define (cpu4-fresh-consume!'):text.index('(define (cpu4-fresh-reject!')]
        self.assertLess(consumer.index('(c1-bytes=?'),consumer.index('(cpu4-resume-run!'))
        self.assertEqual(len(calls('cpu4-fresh-consume!','c1-read-file')),2)
        self.assertIn('(c2-policy-c1-subpolicy-internal',consumer)
        self.assertIn('(require transformer.checkpoint_internal)',driver.CALLER.read_text())
        self.assertIn("'(:overwrite? #f)",text)
        self.assertNotIn('cpu4-trainer-restore-internal!',text)
        self.assertNotIn('g3g-',text)
        self.assertTrue(calls('cpu4-fresh-run!','cpu4-resume-cleanup!'))


if __name__=='__main__':unittest.main()
