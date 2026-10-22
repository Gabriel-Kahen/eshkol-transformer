"""Light source/reader/reference checks; never claims compiled acceptance."""
import hashlib
import json
import re
import subprocess
import tempfile
import ast
import os
from tests.e3_cpu4_consumer import host_reader
import unittest
from pathlib import Path
from tests.e3_cpu4_consumer.resources import generate
from tests.e3_cpu4_consumer.observation import parse
from tests.t1.reference import Identity, fingerprint, decode_artifact

ROOT = Path(__file__).resolve().parents[2]

def forms(text):
    out=[];start=None;depth=0;string=False;escape=False;comment=False
    for i,char in enumerate(text):
        if comment:
            if char=='\n':comment=False
        elif string:
            if escape:escape=False
            elif char=='\\':escape=True
            elif char=='"':string=False
        elif char==';':comment=True
        elif char=='"':string=True
        elif char=='(':
            if depth==0:start=i
            depth+=1
        elif char==')':
            depth-=1
            if depth<0:raise ValueError('unbalanced close')
            if depth==0:out.append(text[start:i+1])
    if depth or string:raise ValueError('unterminated source')
    return out

def nested(form):
    yield form
    for child in forms(form[1:-1]):
        yield from nested(child)


def assert_closed_tail(source):
    definitions=forms(source)
    def definition(name):
        matches=[f for f in definitions if f.startswith('(define ('+name+' ')]
        assert len(matches)==1, 'missing/duplicate '+name
        return matches[0]
    closed=definition('cpu4-e3-restore-acquired-closed!')
    wrapper=definition('e3-restore-closed!')
    checker=definition('cpu4-e3-restored-binding?')
    assert '(guard' not in checker and '(guard' not in closed, 'late handler allocation'
    assert '(cpu4-model-binding-check-internal (vector-ref entry 3))' in checker
    positions=[closed.index(x) for x in (
        '(e3-cleanup-preflight!', '(e3-d2-restore-preflight!', '(e3-native-drain',
        '(e3-d2-restore-commit!', '(p1-e3-mode-restore!', '(cpu4-e3-restored-binding?',
        '(e3-native-publish', '(e3-native-finish', '(e3-source-finish!')]
    assert positions==sorted(positions), 'closed transaction ordering'
    branches=[f for f in nested(closed) if f.startswith('(if publish?')]
    assert len(branches)==1
    branch=branches[0]
    assert '(cpu4-e3-restored-binding? entry)' in branch and '(e3-native-publish' in branch
    assert re.search(r'\s#f\)$',branch), 'abort must skip checker/publication'
    guards=[f for f in nested(wrapper) if f.startswith('(guard')]
    assert len(guards)==1, 'one preinstalled publication handler'
    guard=guards[0]
    children=forms(guard[1:-1])
    assert len(children)==2 and children[0].startswith('(caught'), 'handler shape'
    assert ' '.join(children[1].split())=='(cpu4-e3-restore-acquired-closed! frame entry #t)', 'guard must dominate entire closed helper'
    handler=children[0]
    assert '(e3-note-caught-error! entry 12 caught' in handler
    assert '(e3-native-finish' in handler and '(e3-source-finish!' in handler
    assert '(e3-abort!' not in handler and 'restore-' not in handler, 'no recursive restoration'
    outside=wrapper.replace(guard,'PREINSTALLED-GUARD')
    assert '(guard' not in outside
    assert outside.count('(cpu4-e3-restore-acquired-closed! frame entry #f)')==1
    assert '(cpu4-e3-restore-acquired-closed! frame entry #t)' not in outside
    abort=definition('e3-abort!')
    assert '(guard' not in abort and '(e3-restore-closed! frame #f)' in abort


class SourceTests(unittest.TestCase):
    def test_exact_boundary_forms_and_original_sources(self):
        pins=json.loads((ROOT/'native/e3_cpu4_boundary_forms.json').read_text())
        copied=forms((ROOT/'native/e3_cpu4_boundary.esk').read_text())
        self.assertEqual(len(copied),7)
        for row,actual in zip(pins['forms'],copied):
            raw=(ROOT/row['source']).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(),row['source_sha256'])
            original=subprocess.check_output(['git','show',pins['base_commit']+':'+row['source']],cwd=ROOT)
            self.assertEqual(raw,original)
            expected=next(f for f in forms(raw.decode()) if f.startswith(row['marker']))
            self.assertEqual(actual,expected)
            self.assertEqual(hashlib.sha256(actual.encode()).hexdigest(),row['form_sha256'])

    def test_actual_scheme_reader(self):
        receipt=os.environ.get('CPU4_CONSUMER_HOST_READER_RECEIPT')
        if receipt:
            host_reader.validate(json.loads(Path(receipt).read_text()))
            print('HOST GUILE VERIFIED / GUEST GUILE NOT RUN')
        else:
            result=subprocess.run(['guile',*host_reader.ARGUMENTS],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual((result.returncode,result.stdout,result.stderr),(0,'',''))

    def test_closed_native_tuple_and_root(self):
        names=(ROOT/'native/e3_cpu4_consumer_native_sources.txt').read_text().splitlines()
        self.assertEqual(len(names),17)
        self.assertEqual(len(names),len(set(names)))
        self.assertIn('src/eshkol_transformer/m3_call_f32_integration.c',names)
        self.assertIn('native/e3_d2_native.c',names)
        self.assertIn('src/eshkol_transformer/e3_cpu4_frame.c',names)
        self.assertFalse(set(names)&{'native/f32_tensor.c','native/i64_tensor.c','native/d2_native.c','src/eshkol_transformer/m3_model.c','src/eshkol_transformer/m3t_f32_integration.c','src/eshkol_transformer/e3_frame.c'})
        root=(ROOT/'native/e3_cpu4_private_root.esk').read_text()
        self.assertNotRegex(root,r'\(load "(?:m3t_transport_extension|m3_model_extension|m3_schedule|e3_private_root)')
        self.assertIn('(load "cpu4_model_binding_extension.esk")',root)

    def test_real_raw_v256_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            a=Path(directory)/'a';b=Path(directory)/'b';generate(a);generate(b)
            for path in a.rglob('*'):
                if path.is_file():self.assertEqual(path.read_bytes(),(b/path.relative_to(a)).read_bytes())
            parsed=decode_artifact((a/'tokenizer.tsv').read_bytes())
            self.assertEqual(parsed.identity,Identity())
            self.assertEqual(parsed.fingerprint,fingerprint(Identity()))
            ref=json.loads((a/'reference.json').read_text())
            self.assertEqual((ref['vocabulary'],ref['remaining_tokens'],ref['remaining_batches']),(256,8,2))

    def test_closed_restore_tail_and_order(self):
        source=(ROOT/'native/e3_cpu4_private_extension.esk').read_text()
        assert_closed_tail(source)
        tail=(ROOT/'internal/e3/lib/e3_cpu4_d2_restore.esk').read_text()
        commit=next(f for f in forms(tail) if f.startswith('(define (e3-d2-restore-commit!'))
        self.assertNotRegex(commit,r'(?:cursor-restore|dataset-seek|vector-set! state 8)')
        self.assertIn('(vector-set! core 15 (vector-ref record 7))',commit)

    def test_old_late_guard_and_unguarded_prefix_rejected(self):
        source=(ROOT/'native/e3_cpu4_private_extension.esk').read_text()
        assert_closed_tail(source)
        checker=next(f for f in forms(source) if f.startswith('(define (cpu4-e3-restored-binding?'))
        old_checker='(define (cpu4-e3-restored-binding? entry) (guard (caught (#t #f)) (cpu4-model-binding-check-internal (vector-ref entry 3)) #t))'
        with self.assertRaises(AssertionError):assert_closed_tail(source.replace(checker,old_checker))
        wrapper=next(f for f in forms(source) if f.startswith('(define (e3-restore-closed!'))
        guard=next(f for f in nested(wrapper) if f.startswith('(guard'))
        with self.assertRaises(AssertionError):
            assert_closed_tail(source.replace(guard,'(cpu4-e3-restore-acquired-closed! frame entry #t)'))
        historical=subprocess.check_output(['git','show','59efa27f1c99995d2995c74679a80515e20adbf7:native/e3_cpu4_private_extension.esk'],cwd=ROOT,text=True)
        with self.assertRaises(AssertionError):assert_closed_tail(historical)

    def test_host_reader_receipt_rejects_tamper_head_files_and_failed_reader(self):
        receipt={**host_reader.identity(),'reader':{'executable':'/host/guile',
                 'sha256':'a'*64,'version':'guile (GNU Guile) source test fixture',
                 'arguments':host_reader.ARGUMENTS,'exit':0,'stdout':'','stderr':''}}
        host_reader.validate(receipt)
        for field in ('head','tree'):
            bad=json.loads(json.dumps(receipt));bad[field]='0'*40
            with self.assertRaises(ValueError):host_reader.validate(bad)
        for field in ('sources','code'):
            bad=json.loads(json.dumps(receipt));bad[field][next(iter(bad[field]))]='0'*64
            with self.assertRaises(ValueError):host_reader.validate(bad)
        for field,value in (('exit',37),('stdout','noise'),('stderr','warning'),
                            ('arguments',[]),('sha256','bad')):
            bad=json.loads(json.dumps(receipt));bad['reader'][field]=value
            with self.assertRaises(ValueError):host_reader.validate(bad)
        bad=json.loads(json.dumps(receipt));bad['extra']='tamper'
        with self.assertRaises(ValueError):host_reader.validate(bad)

    def test_exact_transcript_accepts_words_rejects_noise(self):
        valid='CPU4-CONSUMER-REPORT #(1085276160 1090519040 1132462080 0 8 2)\nCPU4-CONSUMER-PASS checks=99\n'
        self.assertEqual(parse(valid)['physical_tokens'],8)
        for bad in (valid+'noise\n','noise\n'+valid,valid.replace(' 8 2)',' 7 2)'),
                    valid.replace('1085276160','4294967296'),valid.replace('1090519040','2139095040'),
                    valid.replace('checks=99','checks=0'),valid.replace('REPORT','PASS')):
            with self.assertRaises(ValueError):parse(bad)

    def test_python310_syntax_and_bash(self):
        for name in ('resources.py','test_source.py','observation.py','host_reader.py'):
            ast.parse((ROOT/'tests/e3_cpu4_consumer'/name).read_text(),feature_version=(3,10))
        subprocess.run(['bash','-n','scripts/test-e3-cpu4-consumer.sh'],cwd=ROOT,check=True)

    def test_actual_exit_seal_both_statuses(self):
        script=(ROOT/'scripts/test-e3-cpu4-consumer.sh').read_text()
        seal=script.split('seal() {',1)[1].split('\ntrap seal EXIT',1)[0]
        for status in (0,37):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);(root/'nested').mkdir()
                (root/'nested/data').write_bytes(b'hello'*500000)
                (root/'empty').write_bytes(b'')
                result=subprocess.run(['bash','-c','out=$1\nseal() {'+seal+'\ntrap seal EXIT\nexit '+str(status),'bash',directory],capture_output=True)
                self.assertEqual((result.returncode,result.stdout,result.stderr),(status,b'',b''))
                self.assertEqual((root/'gate.exit').read_text(),str(status)+'\n')
                rows=dict(line.split('  ',1)[::-1] for line in (root/'SHA256SUMS').read_text().splitlines())
                self.assertEqual(set(rows),{'nested/data','empty','gate.exit'})
                for name,digest in rows.items():self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(),digest)

    def test_sanitizer_helpers_unchanged(self):
        subprocess.run(['sha256sum','--quiet','-c','tests/e3_private/asan_ir_sources.sha256'],cwd=ROOT,check=True)

if __name__=='__main__':unittest.main()
