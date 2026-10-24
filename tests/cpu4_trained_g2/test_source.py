"""Independent AST projection/mutation controls; no Eshkol/native execution."""
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import unittest
from tests.cpu4_trainer_control.test_source_contract import parse, walk
from tests.cpuv1_resource.projection import off, tokens
from tests.cpuv1_resource.test_source import OWNERS as RESOURCE_OWNERS

ROOT = Path(__file__).resolve().parents[2]
PIN = 'b8b717b94a3ca59d0f279b909e5b3c2f80bfcffa'
RESOURCE_PIN = '4dfdec3eb3af12a90c3d3124239cb7e741d50429'
RESOURCE_ADDITIONS = {'native/cpuv1_resource_' + name for name in '''
a2.inc c2.inc c2_reader.inc c4.inc collect.c d2.inc e3.inc events.c
f32.inc generation.inc i2.inc i64.inc inspector.inc io.inc k1.inc m3.inc
o2.inc p1.inc private.h runtime.cpp t1.inc training.inc'''.split()}
HERE = ROOT / 'tests/cpu4_trained_g2'
GENERATE = ['cpu4-trained-g2-generate!', 'resolved', 'tokenizer', 'train-a', 'train-b',
            'heldout', 'factory', 'b-record', 'final-b', 'generation']
FINISH = ['cpu4-trained-g2-finish!', 'generation', 'tokenizer', 'train-a', 'train-b', 'heldout', 'final-b']


def functions(path):
    return {f[1][0]: f for f in parse(path.read_text()) if f[0] == 'define' and isinstance(f[1], list)}


def project(core):
    """Whitelist exact syntactic insertions, including their sibling seam."""
    form = copy.deepcopy(core)
    expected = ['cpu4-trained-g2-core!', 'resolved', 'tokenizer', 'train-a', 'train-b',
                'heldout', 'path', 'policy', 'report', 'mode']
    if form[1] != expected or form[2] != ['cpu4-trained-g2-mode', 'mode']:
        raise ValueError('wrong closed core entry')
    form[1] = ['cpu4-heldout-resume-runtime!'] + expected[1:-1]
    del form[2]
    let = form[3]
    if let[0] != 'let' or let[1][-1] != ['generation', ['cpu4-trained-g2-ledger', 'mode']]:
        raise ValueError('generation ledger not preallocated at original entry')
    let[1].pop()
    if let[2] != ['vector-set!', 'cpu4-trained-g2-session', '0', 'generation']:
        raise ValueError('generation ledger not rooted before original body')
    del let[2]
    seen = {'failure': 0, 'generation': 0, 'finish': 0}
    def transform(node):
        if not isinstance(node, list):
            return node
        if node == ['cpu4-trained-g2-failure-cleanup!', 'ledger', 'generation']:
            seen['failure'] += 1
            return ['cpu4-resume-cleanup!', 'ledger']
        result = []
        for index, child in enumerate(node):
            if child == GENERATE:
                if node[index-1] != ['newline'] or node[index+1] != ['cpu4-resume-cleanup!', 'ledger']:
                    raise ValueError('G2 must follow complete record emission and precede original cleanup')
                seen['generation'] += 1
            elif child == FINISH:
                if node[index-1] != ['newline'] or node[index+1] != '#t':
                    raise ValueError('finish must follow original PASS before true return')
                seen['finish'] += 1
            else:
                result.append(transform(child))
        return result
    form = transform(form)
    if seen != {'failure': 1, 'generation': 1, 'finish': 1}:
        raise ValueError('missing/duplicate allowed core insertion')
    return form


def verify(core, original):
    if project(core) != original:
        raise ValueError('complete original trajectory/criteria/cleanup AST differs')


class Source(unittest.TestCase):
    def setUp(self):
        self.original = functions(ROOT/'tests/cpu4_resume/body.esk')['cpu4-heldout-resume-runtime!']
        self.core = functions(HERE/'core.esk')['cpu4-trained-g2-core!']
        self.generation = functions(HERE/'generation.esk')

    def test_plain_all_files_byte_identical_and_complete_AST_projection(self):
        for path in sorted((ROOT/'tests/cpu4_resume').iterdir()):
            if path.is_file() and path.name not in ('DEPENDENCIES.sha256', 'test_source.py') and path.suffix != '.pyc':
                pinned = subprocess.check_output(['git','show',PIN+':'+str(path.relative_to(ROOT))],cwd=ROOT)
                self.assertEqual(path.read_bytes(), pinned, path)
        verify(self.core,self.original)

    def test_projection_rejects_trajectory_range_criteria_cleanup_and_seam_mutations(self):
        mutations = []
        for target, slot, value in [('cpu4-resume-run!', 3, '249'),
                                    ('cpu4-resume-copy=?', 1, 'at0'),
                                    ('cpu4-resume-cleanup!', 1, 'generation')]:
            bad=copy.deepcopy(self.core)
            node=next(n for n in walk(bad) if n and n[0]==target)
            node[slot]=value;mutations.append(bad)
        bad=copy.deepcopy(self.core)
        node=next(n for n in walk(bad) if n and n[0]=='=' and n[1:] == ['cpu4-resume-steps','375'])
        node[2]='374';mutations.append(bad)
        bad=copy.deepcopy(self.core)
        parent=next(n for n in walk(bad) if GENERATE in n)
        index=parent.index(GENERATE);parent[index],parent[index+1]=parent[index+1],parent[index]
        mutations.append(bad)
        for bad in mutations:
            with self.assertRaises(ValueError):verify(bad,self.original)

    def test_closed_modes_single_handoff_generate_exact_budget_and_release_order(self):
        forms=parse((HERE/'generation.esk').read_text())
        self.assertTrue(all(f[0] in ('define','extern') for f in forms))
        calls=lambda f,name:[n for n in walk(f[2:]) if n and n[0]==name]
        gen=self.generation['cpu4-trained-g2-generate!']
        for name in ('cpu4-checkpoint-factory-generator-handoff-internal!','g3g-public-generator-generate',
                     't1-wave1-tokenizer-byte','t1-wave1-tokenizer-encode','g3c4-public-generation-input-create'):
            self.assertEqual(len(calls(gen,name)),1,name)
        self.assertEqual(sum(len(calls(f,'cpu4-trained-g2-check')) for f in self.generation.values()),19)
        for forbidden in ('fail-after','failpoint','rng-config','model-create-seeded','handoff-retry-internal!',
                          'candidate-adopt','native-capture','restore-internal!'):
            self.assertNotIn(forbidden,(HERE/'generation.esk').read_text())
        release=self.generation['cpu4-trained-g2-release!']
        order=[n[0] for n in walk(release[2:]) if n and isinstance(n[0],str) and n[0].startswith('g3g-public-')]
        self.assertEqual(order,['g3g-public-generation-output-release']+
            ['g3g-public-generation-tensor-release']*3+['g3g-public-generation-rng-release',
             'g3g-public-generation-tensor-release','g3g-public-generator-close'])
        for f in forms:
            for n in walk(f):
                if n and n[0]=='if':self.assertIn(len(n),(3,4))
                if len(n)>2 and n[0]=='let' and isinstance(n[1],str):
                    self.assertFalse(calls(n,'guard'))

    def test_frontend_only_closed_mode_and_load_delta_preserves_resources(self):
        original=parse((ROOT/'tests/cpu4_resume/runtime.esk').read_text())
        new=parse((HERE/'runtime.esk').read_text())
        self.assertEqual(new[:5],[['load','cpu4_v1_private_root.esk'],['load','../cpu4_heldout/body.esk'],
            ['load','../cpu4_resume/body.esk'],['load','generation.esk'],['load','core.esk']])
        self.assertEqual(new[5],original[3])  # exact dataset resource bounds
        runtime=copy.deepcopy(new[6]);old=original[4]
        mode_bind=runtime[2][3][1][-1]
        self.assertEqual(mode_bind[0],'mode')
        branches=mode_bind[1]
        self.assertEqual(branches[0],'cond')
        self.assertEqual([b[1] for b in branches[1:3]],[['quote','two-token'],['quote','first-eos']])
        self.assertEqual(branches[3][0],'else')
        runtime[2][3][1].pop()
        bound=runtime[2][2][1][1][1]
        self.assertEqual(bound,['i2-bounded-list-length','arguments','6'])
        bound[2]='5';runtime[2][2][1][1][2]='4'
        runtime[2][2][2][3]='expected train-directory heldout-directory checkpoint-output'
        call=runtime[2][3][2]
        self.assertEqual(call[0],'cpu4-trained-g2-core!');self.assertEqual(call[-1],'mode')
        call[0]='cpu4-heldout-resume-runtime!';call.pop()
        self.assertEqual(runtime,old)
        self.assertEqual(new[7],original[5])  # actual error guard and D2 close remain unchanged

    def test_actual_dependencies_and_no_production_change(self):
        for row in (HERE/'DEPENDENCIES.sha256').read_text().splitlines():
            digest,path=row.split('  ');self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),digest,path)
            if path == 'tests/cpu4_heldout/inspect.c':
                pinned = subprocess.check_output(['git','show',PIN+':'+path],cwd=ROOT)
                self.assertEqual(tokens(off((ROOT/path).read_text())),tokens(pinned.decode()))
            elif path != 'tests/cpu4_trained_g2/normal_driver.py':
                pinned = subprocess.check_output(['git','show',PIN+':'+path],cwd=ROOT)
                self.assertEqual(hashlib.sha256(pinned).hexdigest(),digest,path)
        changes = subprocess.check_output(['git','diff','--name-only',PIN,'--','native','src','internal','lib','include','tests/cpu4_resume'],cwd=ROOT,text=True).splitlines()
        cli3 = {'native/cpu4_cli_package_bridge.c', 'native/cpu4_cli_report_extension.esk',
                'src/eshkol_transformer/cpu4_cli_scalar_private.h'}
        metadata = {'native/cpu4_v1_source_members.sha256',
                    'tests/cpu4_resume/DEPENDENCIES.sha256', 'tests/cpu4_resume/test_source.py'}
        for path in changes:
            current = (ROOT/path).read_bytes()
            if path in metadata:
                continue  # separately checked by manifest and source-contract suites
            if path in cli3:
                reviewed = subprocess.check_output(['git','show','ab654852:'+path],cwd=ROOT)
                self.assertEqual(current,reviewed,path)
            elif path in RESOURCE_ADDITIONS:
                pinned = subprocess.check_output(['git','show',RESOURCE_PIN+':'+path],cwd=ROOT)
                self.assertEqual(current,pinned,path)
            elif path in RESOURCE_OWNERS:
                pinned = subprocess.check_output(['git','show',PIN+':'+path],cwd=ROOT)
                self.assertEqual(tokens(off(current.decode())),tokens(pinned.decode()),path)
            else:
                pinned = subprocess.check_output(['git','show',PIN+':'+path],cwd=ROOT)
                self.assertEqual(current,pinned,path)
        include_dirs=json.loads((ROOT/'native/cpu4_v1_private_inputs.json').read_text())['source_include_dirs']
        seen=set()
        def closure(path):
            if path in seen:return ''
            seen.add(path);text=path.read_text()
            for f in parse(text):
                if f[0]=='load':
                    child=next((ROOT/d/f[1] for d in include_dirs if (ROOT/d/f[1]).is_file()),None)
                    self.assertIsNotNone(child,f[1]);text+='\n'+closure(child)
            return text
        source=closure(ROOT/'native/cpu4_v1_private_root.esk')
        source+='\n'+(ROOT/'tests/cpu4_heldout/body.esk').read_text()+'\n'+(ROOT/'tests/cpu4_resume/body.esk').read_text()
        own=set(self.generation)|{'cpu4-trained-g2-count','cpu4-trained-g2-core!'}
        for f in list(self.generation.values())+[self.core]:
            for n in walk(f[2:]):
                if n and isinstance(n[0],str) and n[0].startswith(('cpu4-','g3c4-','g3g-','t1-wave1-')) and n[0] not in own:
                    self.assertRegex(source,r'\((?:define(?:-syntax)?\s+\(?|extern\s+\w+\s+)'+re.escape(n[0])+r'(?=[\s)])')


if __name__=='__main__':unittest.main()
