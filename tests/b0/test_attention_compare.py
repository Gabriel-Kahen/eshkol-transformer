"""Comparison schema/protocol negatives and native measured-path parity."""
import copy
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

if __package__:
    from . import attention_compare as compare
else:
    import attention_compare as compare


class ComparisonFormatTests(unittest.TestCase):
    def test_frozen_baseline_is_checked_without_git_history(self):
        frozen = compare.frozen_baseline_source()
        self.assertEqual(compare.hashlib.sha256(frozen).hexdigest(),compare.BASELINE_SHA256)
        with tempfile.TemporaryDirectory() as temporary:
            altered = Path(temporary)/'baseline.c'
            altered.write_bytes(frozen+b'\n')
            with mock.patch.object(compare,'BASELINE_SOURCE',altered):
                with self.assertRaisesRegex(compare.codec.BenchmarkFormatError,'frozen baseline source mismatch'):
                    compare.frozen_baseline_source()

    def test_definition_isolated_and_balanced(self):
        path = compare.ROOT/'benchmarks/attention_compare_v1.json'
        self.assertEqual(compare.load_definition(path),compare.CONTRACT)
        self.assertEqual(compare.encode_definition(compare.CONTRACT),path.read_bytes())
        for loader in [compare.base.load_definition,compare.codec.load_definition]:
            with self.assertRaises(compare.codec.BenchmarkFormatError):
                loader(path)
        for name,value in [('pairs',3),('rss_pairs',3),('pairs',True),('iterations',0),
                ('warmup',-1),('device','cuda:0'),('dtype','f64'),('shapes',[[1,2,2,1,4,1]]),
                ('input_recipe','other'),('operations',['forward']),('scopes',['invoke']),
                ('providers',list(reversed(compare.PROVIDERS))),('unknown',0)]:
            with self.subTest(name=name,value=value),self.assertRaises(compare.codec.BenchmarkFormatError):
                compare.encode_definition(dict(compare.CONTRACT,**{name:value}))

    def test_envelope_rejects_noncanonical_corrupt_duplicate_version(self):
        data = compare.encode_definition(compare.CONTRACT)
        for malformed in [b' '+data,data.replace(b'"pairs":12',b'"pairs":14'),
                data.replace(b'"version":1',b'"version":2'),
                data.replace(b'"version":1',b'"version":1,"version":1')]:
            with self.subTest(data=malformed[-40:]),self.assertRaises(compare.codec.BenchmarkFormatError):
                compare.load_definition(malformed)

    def test_joint_definition_is_opt_in_and_ordered(self):
        path = compare.ROOT/'benchmarks/attention_compare_v1_joint.json'
        self.assertEqual(compare.load_definition(path), compare.JOINT_CONTRACT)
        self.assertEqual(compare.encode_definition(compare.JOINT_CONTRACT), path.read_bytes())
        for definition in [dict(compare.CONTRACT, operations=['forward','backward','joint']),
                           dict(compare.JOINT_CONTRACT, operations=['joint','forward','backward']),
                           dict(compare.JOINT_CONTRACT, operations=['joint'])]:
            with self.assertRaises(compare.codec.BenchmarkFormatError):
                compare.encode_definition(definition)

    def test_native_protocol_parity_order_types_and_nonfinite(self):
        samples = [{'pair':p,'slot':s,'provider':compare.PROVIDERS[(p%2)^s],
            'elapsed_ns':10,'output_bits':'3f80000040000000'} for p in range(2) for s in range(2)]
        protocol = {'version':1,'samples':samples}
        parsed,bits = compare.native_output(json.dumps(protocol).encode(),compare.base.SHAPES[0],'forward',2)
        self.assertEqual(len(parsed),4)
        self.assertEqual(bits,samples[0]['output_bits'])
        for mutation in [lambda x:x.update(version=True),lambda x:x['samples'].pop(),
                lambda x:x['samples'][1].update(output_bits='4000000040000000'),
                lambda x:x['samples'][0].update(output_bits='7f80000040000000'),
                lambda x:x['samples'][0].update(output_bits='0000000000000000'),
                lambda x:x['samples'][0].update(output_bits='3f800000'),
                lambda x:x['samples'][0].update(provider='fallback'),
                lambda x:x['samples'][0].update(pair=True),
                lambda x:x['samples'][0].update(elapsed_ns=0),
                lambda x:x['samples'][2].update(slot=1)]:
            value = copy.deepcopy(protocol)
            mutation(value)
            with self.subTest(mutation=mutation),self.assertRaises(compare.codec.BenchmarkFormatError):
                compare.native_output(json.dumps(value).encode(),compare.base.SHAPES[0],'forward',2)
        with self.assertRaises(compare.codec.BenchmarkFormatError):
            compare.native_output(b'{"version":1,"version":1,"samples":[]}',compare.base.SHAPES[0],'forward',2)

    def test_joint_witness_requires_both_finite_output_parts(self):
        shape = compare.base.SHAPES[0]
        y = '3f80000040000000'
        gradients = '3f800000' * 6
        bits = y + gradients
        self.assertEqual(compare.witness(bits, shape, 'joint'),
                         compare.hashlib.sha256(bytes.fromhex(bits)).hexdigest())
        for invalid in [bits[8:], bits + y, '00000000' * 2 + gradients,
                        y + '00000000' * 6, y + '7f800000' + gradients[8:], None]:
            with self.subTest(bits=invalid), self.assertRaises(compare.codec.BenchmarkFormatError):
                compare.witness(invalid, shape, 'joint')

    def test_support_label_requires_observed_environment_and_declared_image(self):
        env = {'os_release':'ID=ubuntu\nVERSION_ID="22.04"\n','architecture':'x86_64'}
        cc = {'version':'Ubuntu clang version 21.1.8 (source)'}
        self.assertEqual(compare.support_status(env,cc,compare.IMAGE),'supported-native-component')
        for environment,compiler,image in [(env,cc,None),(dict(env,architecture='aarch64'),cc,compare.IMAGE),
                (dict(env,os_release='ID=arch\n'),cc,compare.IMAGE),(env,{'version':'clang version 20.1.0'},compare.IMAGE)]:
            self.assertEqual(compare.support_status(environment,compiler,image),'compatibility-only')

    def test_ratio_analysis_keeps_order_variation(self):
        samples = [{'provider':compare.PROVIDERS[p], 'elapsed_ns':t}
                   for p,t in [(0,100),(1,50),(1,200),(0,100)]]
        summary = compare.summarize(samples)
        self.assertEqual(summary['candidate_over_baseline'],{'min':'0.500000','median':'0.500000','max':'2.000000'})
        self.assertEqual(summary['total_candidate_over_baseline'],'1.250000')
        self.assertEqual(summary['ab_candidate_over_baseline']['median'],'0.500000')
        self.assertEqual(summary['ba_candidate_over_baseline']['median'],'2.000000')


@unittest.skipUnless(platform.system() == 'Linux' and shutil.which('clang'),'Linux/clang required')
class NativeComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Explicit /home scratch even when caller forgot TMPDIR; no /tmp fixtures.
        scratch = compare.ROOT/'build/benchmarks/compare-test-tmp'
        scratch.mkdir(parents=True,exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(dir=scratch)
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        cls.artifacts = compare.build(os.environ.get('CC','clang'),cls.directory)
        cls.definition = dict(compare.CONTRACT,warmup=1,iterations=3,pairs=2,rss_pairs=2)
        cls.definition_path = cls.directory/'definition.json'
        cls.definition_path.write_bytes(compare.encode_definition(cls.definition))
        cls.report_path = cls.directory/'report.json'
        with mock.patch.object(compare,'build',return_value=cls.artifacts):
            compare.run(cls.definition_path,cls.report_path,'clang',cls.directory,None)
        cls.report = compare.load_report(cls.report_path)

    def test_real_all_shapes_both_scopes_forward_backward_and_rss(self):
        self.assertEqual(len(self.report['volatile']['cases']),40)
        for case in self.report['volatile']['cases']:
            self.assertEqual(len(case['samples']),4)
            self.assertEqual(len(case['rss_samples']),4)
            for sample in case['samples']+case['rss_samples']:
                self.assertEqual(sample['output_sha256'],case['output_sha256'])
                self.assertGreater(sample['elapsed_ns'],0)
        self.assertEqual(self.report['volatile']['support_status'],'compatibility-only')
        self.assertEqual(self.report['stable']['baseline']['source_sha256'],compare.BASELINE_SHA256)

    def test_real_known_t1_output_and_cli_negatives(self):
        target = self.artifacts[0]
        output = subprocess.check_output([str(target),'0','forward','invoke','0','1','2','paired'])
        samples,bits = compare.native_output(output,compare.base.SHAPES[0],'forward',2)
        self.assertEqual(bits,'bea000003e000000')  # [-5/16, 2/16], independently known T=1
        self.assertEqual(len(samples),4)
        valid = ['0','forward','dispatch','0','1','2','paired']
        for position,value in [(0,'10'),(0,'-1'),(0,'0x'),(1,'rope'),(2,'fake'),
                (3,'-1'),(4,'0'),(4,'1000001'),(5,'3'),(5,'0'),(6,'auto')]:
            args = valid.copy()
            args[position] = value
            result = subprocess.run([str(target),*args],capture_output=True)
            with self.subTest(args=args):
                self.assertNotEqual(result.returncode,0)
                self.assertEqual(result.stdout,b'')
                self.assertIn(b'attention benchmark:',result.stderr)

    def test_real_joint_report_matches_both_standalone_witnesses(self):
        definition = dict(compare.JOINT_CONTRACT, warmup=1, iterations=3, pairs=2, rss_pairs=2)
        definition_path = self.directory/'joint-definition.json'
        definition_path.write_bytes(compare.encode_definition(definition))
        report_path = self.directory/'joint-report.json'
        with mock.patch.object(compare, 'build', return_value=self.artifacts):
            compare.run(definition_path, report_path, 'clang', self.directory, None)
        report = compare.load_report(report_path)
        cases = report['volatile']['cases']
        self.assertEqual(len(cases), 60)
        self.assertEqual(len(report['volatile']['aggregate']), 6)
        for index, ordinary in enumerate(self.report['volatile']['cases']):
            combined = cases[(index // 4) * 6 + index % 4]
            self.assertEqual(ordinary['output_bits'], combined['output_bits'])
        for index in range(0, 60, 6):
            for slot in range(2):
                forward, backward, joint = cases[index + slot:index + 6:2]
                self.assertEqual(joint['output_bits'], forward['output_bits'] + backward['output_bits'])

        joint = cases[4]
        joint['output_bits'] = '3f800000' + joint['output_bits'][8:]
        joint['output_sha256'] = compare.witness(joint['output_bits'], joint['shape'], 'joint')
        for sample in joint['samples'] + joint['rss_samples']:
            sample['output_sha256'] = joint['output_sha256']
        with self.assertRaisesRegex(compare.codec.BenchmarkFormatError, 'joint/standalone parity mismatch'):
            compare.encode_report(report)

    def test_report_schema_rejects_parity_timing_provenance_and_summary_forgery(self):
        mutations = [lambda r:r['stable'].update(definition_sha256='0'*64),
            lambda r:r['stable']['baseline'].update(source_sha256='0'*64),
            lambda r:r['stable']['project'].update(commit=1234567890123456789012345678901234567890),
            lambda r:r['stable']['sources'].update(extra='0'*64),
            lambda r:r['stable']['compiler'].update(flags=[]),
            lambda r:r['volatile'].update(support_status='supported'),
            lambda r:r['volatile']['environment'].update(cpu_affinity=[True]),
            lambda r:r['volatile']['cases'].pop(),
            lambda r:r['volatile']['cases'][0]['samples'][0].update(pair=True),
            lambda r:r['volatile']['cases'][0]['samples'][0].update(output_sha256='0'*64),
            lambda r:r['volatile']['cases'][0]['samples'][0].update(elapsed_ns=True),
            lambda r:r['volatile']['cases'][0].update(output_bits='7f80000040000000'),
            lambda r:r['volatile']['cases'][0]['rss_samples'][0].update(process_elapsed_ns=1),
            lambda r:r['volatile']['cases'][0]['rss_samples'][1].update(provider=compare.PROVIDERS[0]),
            lambda r:r['volatile']['cases'][0]['summary'].update(total_candidate_over_baseline='0.1'),
            lambda r:r['volatile']['cases'][0]['rss_summary'][compare.PROVIDERS[0]].update(min=1),
            lambda r:r['volatile']['aggregate'][0].update(candidate_over_baseline='0.1')]
        for mutation in mutations:
            report = copy.deepcopy(self.report)
            mutation(report)
            with self.subTest(mutation=mutation),self.assertRaises(compare.codec.BenchmarkFormatError):
                compare.encode_report(report)
        encoded = compare.encode_report(self.report)
        with self.assertRaises(compare.codec.BenchmarkFormatError):
            compare.load_report(encoded.replace(b'"compatibility-only"',b'"supported"'))

    def test_failure_preserves_existing_report(self):
        path = self.directory/'preserved.json'
        path.write_bytes(b'previous report')
        with mock.patch.object(compare,'build',return_value=self.artifacts),mock.patch.object(compare,'measure',side_effect=compare.codec.BenchmarkFormatError('timed out')):
            with self.assertRaises(compare.codec.BenchmarkFormatError):
                compare.run(self.definition_path,path,'clang',self.directory,None)
        self.assertEqual(path.read_bytes(),b'previous report')


if __name__ == '__main__':
    unittest.main()
