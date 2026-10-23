"""Finding-specific emitted-Scheme predicate tests on host identity graphs.

The interpreter executes the generated predicates/helpers. Native FFI leaves use
explicit reference maps solely to test crosslink dispatch and rejection. This is
not native, runtime, compiler, AOT promotion, or allocation-free execution proof;
producer_unit.cpp and tiny_genuine.esk carry the prepared genuine gates.
"""
from pathlib import Path
import unittest
from tests.cpu4_trainer_control.test_source_contract import parse,walk
from tests.cpuv1_resource import generate_scheme as g
from tests.cpuv1_resource.scheme_schema import SCHEMAS

class Vec(list):pass

class Host:
    def __init__(self):
        self.env={
            'eq?':lambda a,b:(a==b if isinstance(a,str) and isinstance(b,str) else a is b),
            'not':lambda x:x is False,'null?':lambda x:isinstance(x,list) and not isinstance(x,Vec) and not x,
            'pair?':lambda x:isinstance(x,list) and not isinstance(x,Vec) and bool(x),
            'car':lambda x:x[0],'cdr':lambda x:x[1:],
            'vector?':lambda x:isinstance(x,Vec),'vector-length':len,'vector-ref':lambda x,i:x[i],
            'integer?':lambda x:type(x) is int,'procedure?':callable,
            '+':lambda *x:sum(x),'=':lambda a,b:a==b,'>':lambda a,b:a>b,'>=':lambda a,b:a>=b,
        }
        # Load actual private quoted constants, not a mirrored phase map.
        sources=((g.ROOT/'tests/cpuv1_resource/helpers.esk').read_text(),
                 g.typed.EXTRA_DEFS[g.P1],g.typed.CANDIDATE_DEFS)
        for source in sources:
            self.constants(source)
    def constants(self,source):
        for form in parse(source):
            if (form[0]=='define' and isinstance(form[1],str)
                    and form[1].startswith(('cpuv1-resource-state-',
                                            'cpuv1-resource-p1-state-',
                                            'cpuv1-resource-candidate-state-',
                                            'cpuv1-resource-literal-'))):
                if len(form)!=3 or form[2][0]!='quote':
                    raise ValueError('private symbol must be a quoted constant')
                self.env[form[1]]=self.ev(form[2])
    def ev(self,n,scope=None):
        env=self.env if scope is None else scope
        if not isinstance(n,list):
            if n=='#t':return True
            if n=='#f':return False
            try:return int(n)
            except ValueError:return env[n]
        if not n:return []
        op=n[0]
        if op=='quote':return n[1]
        if op in ('and','or'):
            value=op=='and'
            for part in n[1:]:
                value=self.ev(part,env)
                if (op=='and' and value is False) or (op=='or' and value is not False):return value
            return value
        if op=='if':return self.ev(n[2] if self.ev(n[1],env) is not False else n[3],env)
        if op=='let':
            local=env|{p[0]:self.ev(p[1],env) for p in n[1]}
            return self.sequence(n[2:],local)
        if op=='cond':
            for row in n[1:]:
                if row[0]=='else' or self.ev(row[0],env) is not False:return self.sequence(row[1:],env)
            return False
        if op=='begin':return self.sequence(n[1:],env)
        return env[op](*(self.ev(x,env) for x in n[1:]))
    def sequence(self,forms,env):
        result=False
        for form in forms:result=self.ev(form,env)
        return result
    def define(self,form):
        name,*args=form[1]
        self.env[name]=lambda *values:self.sequence(form[2:],self.env|dict(zip(args,values)))
    def helpers(self,*names):
        forms=parse((g.ROOT/'tests/cpuv1_resource/helpers.esk').read_text())
        for f in forms:
            if f[0]=='define' and isinstance(f[1],list) and f[1][0] in names:self.define(f)

class Findings(unittest.TestCase):
    def test_candidate_native_cap_owner_phase_and_exact_occurrences(self):
        h=Host();h.helpers('cpuv1-resource-vector?','cpuv1-resource-find-vector',
                          'cpuv1-resource-member?','cpuv1-resource-count-vector')
        cap=object();owner=object();scope=object();context=object()
        token=Vec([False]);token[0]=token
        entry=Vec([object(),'c4-model','live',owner]+[False]*8)
        i2=Vec([False]*11);ticket=Vec([scope]+[False]*5)
        row=Vec([token,'unpublished',cap,owner,entry,i2,ticket,False,False,False,False,False])
        registry=Vec([[row]])
        native={cap:(owner,scope,1,context)};seen=[];states=[];errors=[]
        def bridge(packet,capability,candidate_owner,candidate_scope,phase):
            expected=native.get(capability)
            if expected!=(candidate_owner,candidate_scope,phase,context):return 5
            seen.append(capability);return 0
        h.env.update({'registry':registry,'i2-construction-registry-root':Vec([[i2]]),
            'cpuv1-resource-model-find':lambda shell:entry if shell is entry[0] else False,
            'cpuv1-resource-candidate-link!':bridge,
            'cpuv1-resource-state!':lambda *args:states.append(args),
            'cpuv1-resource-reject!':lambda *args:errors.append(args)})
        h.define(next(f for f in parse(g.typed.CANDIDATE_DEFS) if f[:2]==[
            'define',['cpuv1-resource-candidate-visit','packet','row']]))
        visit=h.env['cpuv1-resource-candidate-visit']
        visit(object(),row);self.assertEqual(seen,[cap]);self.assertEqual(len(states),1)
        for slot,value in [(2,False),(2,object()),(3,object()),(1,'adopted'),(1,'unknown')]:
            old=row[slot];row[slot]=value;seen.clear();states.clear();errors.clear()
            visit(object(),row);self.assertFalse(states);self.assertTrue(errors)
            row[slot]=old
        old_scope=ticket[0];ticket[0]=object();states.clear();errors.clear()
        visit(object(),row);self.assertFalse(states);self.assertTrue(errors);ticket[0]=old_scope
        # Two native candidates in the SAME phase: duplicating one source row
        # and omitting the other balances aggregate counts, but must still fail.
        native[object()]=(object(),object(),1,context)
        for duplicate in [row,Vec(row)]:
            registry[0]=[row,duplicate];states.clear();errors.clear()
            self.assertEqual(len(registry[0]),len(native))
            for candidate in registry[0]:visit(object(),candidate)
            self.assertFalse(states);self.assertEqual(len(errors),2)
        registry[0]=[row]
        for word,phase in [('unpublished',1),('adopted',2),('dead',3)]:
            row[1]=word;entry[2]='dead' if phase==3 else 'live'
            native[cap]=(owner,scope,phase,context);states.clear();errors.clear()
            visit(object(),row);self.assertFalse(errors)
            self.assertEqual(states[0][1:],(64,2 if phase==3 else 1,5,phase))
            old=entry[2];entry[2]='live' if phase==3 else 'dead'
            states.clear();errors.clear();visit(object(),row)
            self.assertFalse(states);self.assertTrue(errors);entry[2]=old

    def test_provider_eight_native_callback_slots_and_omission(self):
        h=Host();h.helpers('cpuv1-resource-vector?','cpuv1-resource-provider-callbacks!',
                          'cpuv1-resource-count-vector')
        context=object();provider=object();callbacks=Vec([object() for _ in range(8)])
        admitted=Vec([False]*10+[callbacks]);registration=Vec(['i2',provider,admitted])
        shell=Vec([provider,'provider',admitted]);seen=[]
        def callback(packet,ctx,owner,token,slot):
            if ctx is not context or owner is not provider or token is not callbacks_original[slot]:return 5
            seen.append(slot);return 0
        callbacks_original=list(callbacks)
        h.env.update({'packet':object(),'row':registration,'native-context':context,
                      'shell-entry':lambda value:shell if value is provider else False,
                      'tensor-provider-registry-root':Vec([[registration]]),
                      'cpuv1-resource-callback-native':callback})
        check=parse(SCHEMAS[2]['check'])[0]
        self.assertTrue(h.ev(check));self.assertEqual(seen,list(range(8)))
        # Real P1 records are provider + eight callbacks; only provider is a
        # Scheme shell/index entry. The old all-record equality rejected this.
        self.assertEqual(1+len(seen),9);self.assertNotEqual(1,9)
        self.assertEqual(len(seen),8*1)
        for bad in [Vec(callbacks[:7]),Vec(callbacks[:7]+[callbacks[0]])]:
            admitted[10]=bad;seen.clear();self.assertFalse(h.ev(check));self.assertLess(len(seen),8)
        admitted[10]=callbacks;h.env['native-context']=object();seen.clear()
        self.assertFalse(h.ev(check));self.assertFalse(seen)

    def test_two_provider_same_or_copied_registration_cannot_omit_other(self):
        h=Host();h.helpers('cpuv1-resource-vector?','cpuv1-resource-provider-callbacks!',
                          'cpuv1-resource-count-vector')
        context=object();providers=[object(),object()]
        callbacks={provider:Vec([object() for _ in range(8)]) for provider in providers}
        admitted={provider:Vec([False]*10+[callbacks[provider]]) for provider in providers}
        registrations=[Vec(['provider',provider,admitted[provider]]) for provider in providers]
        shells={provider:Vec([provider,'provider',admitted[provider]]) for provider in providers}
        root=Vec([registrations]);seen=[]
        def callback(packet,ctx,provider,token,slot):
            if ctx is not context or provider not in callbacks or token is not callbacks[provider][slot]:return 5
            seen.append((provider,slot));return 0
        h.env.update({'packet':object(),'native-context':context,
                      'tensor-provider-registry-root':root,
                      'shell-entry':lambda provider:shells.get(provider,False),
                      'cpuv1-resource-callback-native':callback})
        check=parse(SCHEMAS[2]['check'])[0]
        for registration in registrations:
            h.env['row']=registration;self.assertTrue(h.ev(check))
        self.assertEqual(seen,[(provider,slot) for provider in providers for slot in range(8)])
        # Both source rows can otherwise validate A's exact eight callback
        # slots twice, balancing native A+B's16 callbacks while omitting B.
        for duplicate in [registrations[0],Vec(registrations[0])]:
            root[0]=[registrations[0],duplicate];seen.clear()
            self.assertEqual(len(root[0])*8,sum(map(len,callbacks.values())))
            for registration in root[0]:
                h.env['row']=registration;self.assertFalse(h.ev(check))
            self.assertFalse(seen)

    def test_native_guard_source_and_declared_static_bytes(self):
        p1=(g.ROOT/'native/cpuv1_resource_p1.inc').read_text()
        shell=p1.split('int64_t et_cpuv1_test_p1_shell_v1',1)[1].split('int64_t et_cpuv1_test_p1_links_finish_v1',1)[0]
        self.assertIn('token_identity_integrity(record)',shell)
        self.assertNotIn('!token_integrity(record)',shell)
        self.assertIn('record->origin_pid!=(int64_t)getpid()',shell)
        self.assertIn('context_integrity(record->owner)',shell)
        mutation=p1.split('int64_t et_cpuv1_test_p1_shell_mutation_v1',1)[1]
        self.assertIn('record->token->nonce_lo=nonce;record->origin_pid=pid;',mutation)
        unit=(g.ROOT/'tests/cpuv1_resource/producer_unit.cpp').read_text()
        for name in ['et_p1_private_provider_seal_release_v1',
                     'et_p1_private_callback_identity_create_v1',
                     'et_p1_private_construction_module_create_v1',
                     'et_p1_private_construction_abort_v1',
                     'et_cpuv1_test_p1_shell_mutation_v1']:
            self.assertIn(name,unit)
        callback=p1.split('int64_t et_cpuv1_test_p1_callback_v1',1)[1].split('int64_t et_cpuv1_test_p1_shell_v1',1)[0]
        for text in ['!token_integrity(c)','c->binding!=p','p->owner!=context',
                     '(slot<7?p->callbacks[slot]:p->binding)!=c']:
            self.assertIn(text,callback)
        self.assertIn('scheme_states[3][0]!=packet->scheme_p1[14]',p1)
        self.assertNotIn('scheme_states[3][0]!=packet->rows[ET_R_P1][1].bases',p1)
        c4=(g.ROOT/'native/cpuv1_resource_c4.inc').read_text()
        for text in ['while(p && p!=capability)','while(o && o!=owner)',
                     'p->owner!=o','o->candidate!=p','p->state!=expected',
                     'p->p1_scope!=scope','cpu4_candidate_dead_exact(p)',
                     'packet->scheme_source[64][i+1]']:
            self.assertIn(text,c4)
        runtime=(g.ROOT/'native/cpuv1_resource_runtime.cpp').read_text()
        self.assertIn('sizeof(resource_root)+sizeof(resource_page_bytes)+sizeof(resource_warmed)',runtime)
        events=(g.ROOT/'native/cpuv1_resource_events.c').read_text()
        self.assertIn('et_cpuv1_test_runtime_static_bytes_v1()',events)

    def test_cpp_recipe_hidden_visibility(self):
        import tempfile,json
        from tests.cpuv1_resource.prepare_recipes import prepare
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'fresh';prepare(out,'/pin/cc','/pin/cxx','/pin/source','/pin/build')
            commands=json.loads((out/'native-unit.commands.json').read_text())
            for command in commands:
                if command[0]=='/pin/cxx':self.assertIn('-fvisibility=hidden',command)

if __name__=='__main__':unittest.main()
