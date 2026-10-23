"""Read/projection/mutation controls only; no Scheme/AOT/native execution."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from tests.cpu4_trainer_control.test_source_contract import parse,walk
from tests.cpuv1_resource import generate_scheme as g


class Scheme(unittest.TestCase):
    def test_single_explicit_graph_and_exact_reverse_projection(self):
        with tempfile.TemporaryDirectory() as td:
            receipt=g.generate(Path(td)/'generated')
            p1=[r for r in receipt['units'] if r['path']==g.P1]
            self.assertEqual(len(p1),1)
            self.assertEqual((receipt['semantic_roots'],receipt['p1_lexical_values']),(64,51))
            targets=set()
            for row in receipt['units']:
                for form in parse(Path(row['generated']).read_text()):
                    if form[0] in ('require','load'):
                        self.assertTrue(Path(form[1]).is_absolute())
                        if 'internal/p1/lib/transformer/module.esk' in form[1]:targets.add(form[1])
            self.assertEqual(targets,{p1[0]['generated']})

    def test_original_trusted_vector_and_one_native_context(self):
        original=(g.ROOT/g.P1).read_text();staged,record=g.transform(original,g.P1)
        forms=parse(staged)
        envelope=next(f for f in forms if f[:2]==['define','cpuv1-p1-test-envelope'])
        original_form=next(f for f in parse(original) if f[:2]==['define','p1-trusted-surface'])
        self.assertEqual(envelope[2][-1][1],original_form[2][-1])
        self.assertEqual(len(envelope[2][-1][1])-1,75)
        self.assertEqual(sum(n[0]=='p1-native-context-create' for n in walk(forms) if n),1)
        closure=envelope[2][-1][2]
        self.assertIn(['shell-registry-current'],list(walk(closure)))
        calls=[n for n in walk(closure) if n and n[0]=='cpuv1-resource-value-into!']
        self.assertEqual(len([n for n in calls if n[-1]=='1']),51)
        self.assertEqual({n[3] for n in calls if n[-1]=='0'},
                         {r['name'] for r in g.ROOTS if r['path']==g.P1})
        self.assertIn('active-record-root',staged)
        self.assertIn('candidate-ticket-root',staged)
        self.assertIn('shell-identity-index',staged)

    def test_inverse_rejects_slot_export_and_alias_changes(self):
        original=(g.ROOT/g.P1).read_text();staged,record=g.transform(original,g.P1)
        for changed in [staged.replace('(vector-ref cpuv1-p1-test-envelope 0)',
                                        '(vector-ref cpuv1-p1-test-envelope 1)',1),
                        staged.replace('p1-trusted-surface '+record['getter'],
                                        'p1-trusted-surface wrong-getter',1),
                        staged.replace('(vector (vector','(vector (vector #f',1)]:
            with self.subTest(change=changed[:50]):
                with self.assertRaises(ValueError):
                    g.reverse_projection(changed,record,{},'__HELPER__')

    def test_helper_has_no_snapshot_constructors_or_hash_materialization(self):
        helper=(g.ROOT/'tests/cpuv1_resource/helpers.esk').read_text()
        forbidden={'cons','vector','make-vector','lambda','guard','hash-table-keys','hash-table-values'}
        for n in walk(parse(helper)):
            if n and isinstance(n[0],str):self.assertNotIn(n[0],forbidden)
        self.assertIn('hash-table-ref',helper)
        self.assertIn('16777216',helper)
        self.assertIn('et_cpuv1_test_p1_shell_v1',helper)
        # External-only insertion has no lexical access to the trusted roots.
        p1=parse((g.ROOT/g.P1).read_text())
        outer={f[1] for f in p1 if f[0]=='define' and isinstance(f[1],str)}
        self.assertNotIn('shell-registry-root',outer)
        self.assertNotIn('native-context',outer)

    def test_native_cannot_synthesize_scheme_complete(self):
        collect=(g.ROOT/'native/cpuv1_resource_collect.c').read_text()
        self.assertNotIn('resource_domain_v1(p,ET_R_SCHEME)',collect)
        events=(g.ROOT/'native/cpuv1_resource_events.c').read_text()
        self.assertIn('i<64;i++)if(!p->details[ET_R_SCHEME][i])',events)
        self.assertIn('i<51;i++)if(!p->scheme_lexical[i][0])',events)
        self.assertIn('p->scheme_classified!=UINT64_MAX',events)


if __name__=='__main__':unittest.main()

class Typed(unittest.TestCase):
    def test_every_selected_root_has_concrete_classifier(self):
        from tests.cpuv1_resource.scheme_schema import SCHEMAS
        self.assertEqual(set(SCHEMAS),set(range(64)))
        for row in g.ROOTS:
            self.assertTrue(g.typed.typed_call(row))
        # Every generic write stays separate from the concrete owning seal.
        events=(g.ROOT/'native/cpuv1_resource_events.c').read_text()
        generic=events.split('int64_t et_cpuv1_test_scheme_value_v1(',1)[1].split('/* Only concrete',1)[0]
        self.assertNotIn('scheme_classified',generic)
        self.assertIn('p->scheme_states[index][0]!=(uint64_t)count',events)
        self.assertIn('et_cpuv1_test_p1_links_finish_v1(p)',events)
        self.assertIn('!p->scheme_candidate_ready',events)

    def test_candidate_same_scope_and_six_slot_inverse(self):
        path=g.typed.CANDIDATE
        original=(g.ROOT/path).read_text();staged,record=g.transform(original,path)
        g.reverse_projection(staged,record,{},'__HELPER__')
        envelope=next(f for f in parse(staged) if f[:2]==['define','cpuv1-candidate-test-envelope'])
        self.assertEqual(len(envelope[2][-1][1])-1,6)
        self.assertIn(['vector-ref','registry','0'],list(walk(envelope[2][-1][2])))
        for changed in [staged.replace('(vector-ref registry 0)','(vector-ref wrong-registry 0)',1),
                        staged.replace('(vector-ref cpuv1-candidate-test-envelope 0)',
                                       '(vector-ref cpuv1-candidate-test-envelope 1)',1)]:
            with self.assertRaises(ValueError):g.reverse_projection(changed,record,{},'__HELPER__')

    def test_boolean_values_and_safe_malformed_shell_branch(self):
        helper=parse((g.ROOT/'tests/cpuv1_resource/helpers.esk').read_text())
        generic=next(f for f in helper if f[:2]==['define',['cpuv1-resource-value-into!','packet','index','value','lexical']])
        self.assertIn(['if','value','1','0'],list(walk(generic)))
        shells=next(f for f in helper if f[0]=='define' and isinstance(f[1],list) and f[1][0]=='cpuv1-resource-p1-shells')
        guard=next(n for n in walk(shells) if n and n[0]=='if')
        # Native token read occurs only in validated branch; reject returns.
        self.assertEqual(guard[1][0],'and')
        self.assertEqual(guard[3],['cpuv1-resource-reject!','packet'])
        self.assertIn('cpuv1-resource-shell-native',str(guard[2]))

    def test_closed_cpu4_schema_mutations(self):
        """Evaluate the emitted pure schema AST on a host identity graph.

        This tests the concrete predicate, not Scheme lowering or native state.
        No CPU4 production operation is replaced or claimed to have executed.
        """
        from tests.cpuv1_resource.scheme_schema import SCHEMAS
        class Vec(list):pass
        def ev(n,env):
            if not isinstance(n,list):
                if n=='#f':return False
                if n=='#t':return True
                try:return int(n)
                except ValueError:return env[n]
            op=n[0]
            if op=='quote':return n[1]
            if op=='and':
                value=True
                for item in n[1:]:
                    value=ev(item,env)
                    if value is False:return False
                return value
            if op=='or':
                for item in n[1:]:
                    value=ev(item,env)
                    if value is not False:return value
                return False
            if op=='if':return ev(n[2] if ev(n[1],env) is not False else n[3],env)
            args=[ev(x,env) for x in n[1:]]
            return env[op](*args)
        tag=object();record=Vec([object()]*22);trainer=Vec([record,object(),object()])
        holder=Vec([[trainer],False]);frame=Vec([False]*20)
        frame[0]=tag;frame[1]=frame;frame[2]=record;frame[3]=trainer;frame[4]='committed';frame[17]=Vec([0]*4)
        env={'row':frame,'cpu4-step-tag':tag,'cpu4-step-frames':Vec([[frame],False]),'cpu4-trainers':holder,
             'cpuv1-resource-vector?':lambda a,b:isinstance(a,Vec) and len(a)==b,
             'vector-ref':lambda a,b:a[b], 'eq?':lambda a,b:(a==b if isinstance(a,str) else a is b),
             'not':lambda a:a is False,'cpuv1-resource-member?':lambda a,b,c:any(a is v for v in b),
             'cpuv1-resource-trainer-record?':lambda a:a is record}
        check=parse(SCHEMAS[24]['check'])[0]
        self.assertTrue(ev(check,env))
        mutations=[(0,object()),(1,Vec(frame)),(2,Vec(record)),(3,Vec(trainer)),(4,'active'),(11,object()),(15,object()),(16,True),(17,Vec([0]*3)),(18,object())]
        for slot,value in mutations:
            original=frame[slot];frame[slot]=value
            self.assertFalse(ev(check,env),(slot,value))
            frame[slot]=original
        env['cpu4-step-frames'][1]=True
        self.assertFalse(ev(check,env))
        env['cpu4-step-frames'][1]=False
        env['row']=Vec([tag])
        self.assertFalse(ev(check,env))

    def test_source_phases_and_lease_roster_are_not_truthy_aliases(self):
        from tests.cpuv1_resource.scheme_schema import SCHEMAS,RAW_NAMES,EXTRA_DEFS
        self.assertEqual(RAW_NAMES[24][5:7],['committed','aborted'])
        self.assertNotEqual(SCHEMAS[24]['raw'],SCHEMAS[24]['state'])
        self.assertEqual(RAW_NAMES[64],['STATIC','unpublished','adopted','dead'])
        roster=EXTRA_DEFS['native/cpu4_model_binding_extension.esk']
        for required in ['cpu4-native-lease-parameter','cpu4-native-lease-identity',
                         'i2-native-g3c4-owner-parameter','(= index 14)',
                         '(eq? leased native)','(eq? identity handle)']:
            self.assertIn(required,roster)
        self.assertIn('cpuv1-resource-binding-roster? row 0',SCHEMAS[21]['check'])

    def test_preparation_never_executes_and_tiny_uses_real_original_lifetime(self):
        from tests.cpuv1_resource.prepare_recipes import prepare
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'recipe'
            receipt=prepare(out,'/pin/cc','/pin/cxx','/pin/fe9','/pin/build')
            self.assertEqual(receipt['execution'],'NOT RUN')
            self.assertEqual(receipt['native_units'],37)
            native=json.loads((out/'native-unit.commands.json').read_text())
            self.assertTrue(any(str(g.ROOT/'tests/cpu4_heldout/inspect.c') in c and '-DET_O2_TESTING' in c for c in native))
            self.assertTrue(any(str(g.ROOT/'tests/cpu4_candidate_children/native_observer.c') in c and '-DET_F32_TENSOR_TESTING' in c for c in native))
            self.assertTrue(any(str(g.ROOT/'src/eshkol_transformer/m3_call_f32_integration.c') in c and '-DET_F32_TENSOR_TESTING' in c for c in native))
            plan=json.loads((out/'tiny-compile.commands.json').read_text())
            self.assertIn(str(out/'generated/tiny.esk'),plan[0])
            self.assertTrue((out/'native-unit.sh').is_file())
            self.assertEqual(len([r for r in receipt['generation']['units'] if r['path']==g.P1]),1)
            with self.assertRaises(ValueError):prepare(out,'cc','cxx','source','build')
        tiny=(g.ROOT/'tests/cpuv1_resource/tiny_genuine.esk').read_text()
        for name in ['cpu4-owned-candidate-create-internal','module-train!',
                     'module-eval!','cpu4-owned-candidate-retire-internal!']:
            self.assertEqual(sum(n[0]==name for n in walk(parse(tiny)) if n),1)
        self.assertNotIn('cpu4-training-step-internal!',tiny)
        self.assertNotIn('cpu4-candidate-adopt-internal!',tiny)
        self.assertIn('indirect closure',tiny)
