"""Source/extraction/checker controls only. Never compile or execute a witness."""
import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
from tests.cpu4_trainer_control.test_source_contract import parse, walk
from .generator import BASE, SOURCE, ORIGINAL, RENAMED, extract, generated, generate
from .check import EXPECTED, check

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'tests/cpu4_binding_witness'
BASE_HASH = '7ff90345a23204a6b182b7d12417350c07cdd9871b1a23b54dd49b62a38dd567'
REPAIRED_HASH = '4cc9d1feb4530da9c12e3224b9978d72bf0f798adaa46931ffc6e878b8ece346'


def retained_dead_contract(forms):
    case = next(f for f in forms if f[:2] == ['define',['cpu4bw-case','mode']])
    nodes = list(walk(case))
    initial = ['cpu4bw-check','fresh owner has no prior binding row',
               ['equal?',['cpu4bw-owner-rows'],['vector','0','0']]]
    retained = ['if',['=','mode','3'],
                ['cpu4bw-check','current published row retained DEAD before retry',
                 ['equal?',['cpu4bw-owner-rows'],['vector','1','1']]]]
    assert initial in nodes, 'fresh owner must have zero prior rows'
    assert retained in nodes, 'current attempted row must remain exactly one DEAD row'
    return initial, retained


class Source(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = subprocess.check_output(['git','show',BASE+':'+SOURCE],cwd=ROOT,text=True)
        cls.body = (HERE/'body.esk').read_text()
        cls.shim = (HERE/'shim.cpp').read_text()
        cls.forms = parse(cls.body)
        cls.functions = {f[1][0]: f for f in cls.forms if f[0]=='define' and isinstance(f[1], list)}

    def test_exact_baseline_constructor_only_header_rename(self):
        self.assertEqual(hashlib.sha256(self.baseline.encode()).hexdigest(), BASE_HASH)
        self.assertEqual(hashlib.sha256((ROOT/SOURCE).read_bytes()).hexdigest(), REPAIRED_HASH)
        for path,digest in {
            'tests/m3cg/handler_failure_shim.cpp': '11c4ee67711e976907eb96d3cd3690be9a579f3a33dfc1e7f119ce3e95c1ebe7',
            'tests/m3cg/handler_failure.esk': '2f8ebed32876a12a664e3307137dce7d333144d5b91787a125f717b972ad98f8',
        }.items():
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),digest)
        before = parse(extract(self.baseline))
        result = parse(generated(self.baseline))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1][0], RENAMED)
        result[0][1][0] = ORIGINAL
        self.assertEqual(result, before)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'baseline.esk'
            generate(ROOT,output)
            self.assertEqual(output.read_text(), generated(self.baseline))
            with self.assertRaises(ValueError): generate(ROOT,output)
        # Strings/comments containing parentheses cannot truncate extraction.
        sample = f'(define ({ORIGINAL} model resolved) ; )\n (display "(text)") #t)\n(define unrelated 1)'
        self.assertEqual(len(parse(extract(sample))), 1)
        with self.assertRaises(ValueError): extract(sample[:sample.index('#t')])

    def test_one_external_identity_root_and_actual_three_cases(self):
        self.assertTrue(all(f[0] in ('define','extern') for f in self.forms))
        main = self.functions['cpu4-binding-witness-run']
        calls = [n for n in walk(main) if n and n[0]=='cpu4bw-case']
        self.assertEqual(calls, [['cpu4bw-case','1'],['cpu4bw-case','2'],['cpu4bw-case','3']])
        case = self.functions['cpu4bw-case']
        heads = [n[0] for n in walk(case) if n and isinstance(n[0],str)]
        for actual in ('cpu4-owned-candidate-create-internal','cpu4-owned-model-internal','module-train!',
                       'cpu4-binding-live-entry','cpu4-model-binding-create-internal',
                       'cpu4-model-binding-check-internal','cpu4-model-binding-release-internal!',
                       'cpu4-owned-candidate-retire-internal!'):
            self.assertIn(actual,heads)
        self.assertNotIn('cpu4-native-lease-enter',self.body)
        self.assertNotIn('training-step',self.body)
        for f in self.forms:
            for n in walk(f):
                if n and n[0]=='if': self.assertIn(len(n),(3,4))
                if len(n)>2 and n[0]=='let' and isinstance(n[1],str):
                    self.assertNotIn(n[1],('values','match','length','map','list','vector'))

    def test_real_capture_delegation_and_restore_before_repaired_retry(self):
        capture = self.functions['cpu4bw-capture']
        heads = [n[0] for n in walk(capture) if n and isinstance(n[0],str)]
        self.assertLess(heads.index('cpu4bw-original-capture'),heads.index('cpu4bw-post'))
        self.assertLess(heads.index('cpu4bw-post'),heads.index('cpu4bw-reserve'))
        self.assertIn(['cpu4bw-reserve','129'],list(walk(capture)))
        case_text = self.body[self.body.index('(define (cpu4bw-case'):self.body.index('(define (cpu4-binding-witness-run')]
        self.assertLess(case_text.index('(vector-set! (vector-ref p1-trusted-surface 74) 0 cpu4bw-original-capture)'),
                        case_text.index('(cpu4-model-binding-create-internal model cpu4bw-config)'))
        self.assertIn(['define','cpu4bw-original-capture',
                       ['vector-ref',['vector-ref','p1-trusted-surface','74'],'0']],self.forms)
        self.assertNotIn('set! cpu4-p1-fixed-capture-internal',self.body)
        baseline_branch = next(n for n in walk(self.functions['cpu4bw-case'])
                               if len(n)>3 and n[:2]==['if',['=','mode','1']])
        self.assertIn('quarantine-baseline',str(baseline_branch[2]))
        self.assertNotIn('quarantine-baseline',str(baseline_branch[3]))
        baseline_text=case_text[case_text.index('(if (= mode 1)'):]
        self.assertLess(baseline_text.index('(cpu4bw-baseline-retry-blocked?)'),baseline_text.index('(let quarantine-baseline'))

    def test_non_tail_occupancy_and_actual_hit_not_pool_bound(self):
        scope = self.functions['cpu4bw-scope']
        nodes = list(walk(scope))
        self.assertIn(['begin',['cpu4bw-scope',['+','depth','1']],['cpu4bw-fresh']],nodes)
        self.assertIn(['<','depth','128'],nodes)
        self.assertIn('preparing && fresh && !armed',self.shim)
        self.assertIn('bytes == sizeof(eshkol_exception_handler_t)',self.shim)
        self.assertIn('failed_mallocs == 1 && create_attempts == 0',self.shim)
        self.assertIn('eshkol_runtime_emergency_raise_v1(5)',self.shim)
        for real in ('__real_malloc','__real_eshkol_push_exception_handler','__real_eshkol_get_raised_value'):
            self.assertIn(real,self.shim)
        for exact in ('value->flags','value->reserved','== condition5','g_current_exception == condition5',
                      'g_exception_handler_stack == expected','mode == 3 ? 2 : 1'):
            self.assertIn(exact,self.shim)

    def test_only_existing_native_lease_wrappers_and_honest_limits(self):
        for name in ('create','prepare_release','commit_release'):
            self.assertIn('__wrap_et_g3c4_private_training_lease_'+name+'_v1',self.shim)
            self.assertIn('__real_et_g3c4_private_training_lease_'+name+'_v1',self.shim)
        self.assertIn('et_g3c4_private_training_lease_check_v1(lease) != 0',self.shim)
        for unsupported in ('relocation=UNPROVEN','retained-bytes=UNMEASURED'):
            self.assertIn(unsupported,self.body)
        self.assertNotIn('free(observed_lease)',self.shim)
        self.assertNotIn('lease_count',self.shim)

    def test_current_retained_dead_row_not_deletion_or_prior_dead_row(self):
        import copy
        initial, retained = retained_dead_contract(self.forms)
        text = self.body[self.body.index('(define (cpu4bw-case'):]
        self.assertLess(text.index('fresh owner has no prior binding row'), text.index('(cpu4bw-prepare'))
        self.assertLess(text.index('current published row retained DEAD before retry'),
                        text.index('(cpu4-model-binding-create-internal model'))
        for mutation in ('delete-check','deletion-is-cleanup','prior-dead-allowed'):
            bad = copy.deepcopy(self.forms)
            target = next(n for f in bad for n in walk(f) if n == (initial if mutation=='prior-dead-allowed' else retained))
            if mutation == 'delete-check': target[:] = ['begin','#t']
            elif mutation == 'deletion-is-cleanup': target[2][2][-1] = ['vector','0','0']
            else: target[-1][-1] = ['vector','1','1']
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError): retained_dead_contract(bad)
        # Execute the ACTUAL new predicate in Guile with explicit owner/row
        # identity doubles; no fe9/native allocation or registry claim follows.
        program = r'''
(define cpu4bw-owner (vector 'owner))
(define cpu4-model-bindings (vector '()))
(call-with-input-file "tests/cpu4_binding_witness/body.esk"
 (lambda (p) (let read-selected () (let ((f (read p)))
  (if (not (eof-object? f))
   (begin (if (and (eq? (car f) 'define) (pair? (cadr f))
                  (eq? (caadr f) 'cpu4bw-owner-rows)) (eval f (current-module)))
          (read-selected)))))))
(define (need condition) (if (not condition) (error "retained DEAD host predicate")))
(define row (vector #f 'pending #f #f cpu4bw-owner))
(need (equal? (cpu4bw-owner-rows) (vector 0 0)))
(vector-set! cpu4-model-bindings 0 (list row))
(need (equal? (cpu4bw-owner-rows) (vector 1 0)))
(vector-set! row 1 'dead)
(need (equal? (cpu4bw-owner-rows) (vector 1 1)))
(need (not (equal? (cpu4bw-owner-rows) (vector 0 0))))
(vector-set! cpu4-model-bindings 0 '())
(need (not (equal? (cpu4bw-owner-rows) (vector 1 1))))
(vector-set! cpu4-model-bindings 0 (list row row))
(need (not (equal? (cpu4bw-owner-rows) (vector 1 1))))
(display "CPU4BW retained-DEAD actual helper host doubles PASS: checks=6\n")
'''
        result = subprocess.run(['guile','--no-auto-compile','-c',program],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout,'CPU4BW retained-DEAD actual helper host doubles PASS: checks=6\n')

    def test_checker_rejects_each_causal_field_and_region_mutation(self):
        self.assertTrue(check(EXPECTED))
        for field in ('case','pushes','condition5-reads','failure-create-attempts','pending',
                      'retry-blocked','retry-create','prepare','commit','dead-check'):
            for line_number in range(3):
                lines=EXPECTED.splitlines()
                parts=lines[line_number].split(' ')
                index=next(i for i,part in enumerate(parts) if part.startswith(field+'='))
                key,value=parts[index].split('='); parts[index]=key+'='+str(int(value)+1)
                lines[line_number]=' '.join(parts)
                with self.subTest(field=field,line=line_number), self.assertRaises(ValueError):
                    check('\n'.join(lines)+'\n')
        for invalid in (EXPECTED.rstrip('\n'),EXPECTED+'extra\n','\n'.join(reversed(EXPECTED.splitlines()))+'\n',
                        EXPECTED.replace('relocation=UNPROVEN','relocation=PROVEN'),EXPECTED.replace('\n','\r\n')):
            with self.assertRaises(ValueError): check(invalid)


if __name__ == '__main__':
    unittest.main()
