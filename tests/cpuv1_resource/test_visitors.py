"""Host identity/projection controls; successor emitted-IR proof is separate."""
import unittest
from tests.cpu4_trainer_control.test_source_contract import parse,walk
from tests.cpuv1_resource import generate_scheme as g
from tests.cpuv1_resource.test_review_fixes import Host


class Visitors(unittest.TestCase):
    def test_each_registry_passes_one_prebound_value(self):
        for path in {r['path'] for r in g.ROOTS}|{g.typed.CANDIDATE}:
            staged,_=g.transform((g.ROOT/path).read_text(),path)
            forms=parse(staged);nodes=list(walk(forms))
            expected={g.typed.visitor_value_name(r['id']):g.typed.visitor_name(r['id'])
                      for r in g.ROOTS if r['path']==path
                      and g.typed.SCHEMAS[r['id']]['mode']=='registry'}
            if path==g.typed.CANDIDATE:
                expected['cpuv1-resource-candidate-visitor']='cpuv1-resource-candidate-visit'
            for value,target in expected.items():
                self.assertEqual(nodes.count(['define',value,target]),1)
                self.assertFalse(any(n[:2]==['set!',value] for n in nodes))
            for node in nodes:
                if node and node[0]=='cpuv1-resource-walk!':
                    self.assertIn(node[4],expected)

    def test_bindings_are_in_original_owning_initialization(self):
        for path in {r['path'] for r in g.ROOTS}|{g.typed.CANDIDATE}:
            staged,record=g.transform((g.ROOT/path).read_text(),path)
            forms=parse(staged)
            if path in (g.P1,g.typed.CANDIDATE):
                name='cpuv1-p1-test-envelope' if path==g.P1 else 'cpuv1-candidate-test-envelope'
                envelope=next(f for f in forms if f[:2]==['define',name])
                initialization=envelope[2][2:-1]
            else:initialization=forms
            all_bindings=[n for n in walk(forms) if n and n[0]=='define'
                          and isinstance(n[1],str)
                          and (n[1].startswith('cpuv1-resource-visitor-')
                               or n[1]=='cpuv1-resource-candidate-visitor')]
            self.assertTrue(all(n in initialization for n in all_bindings))
            g.reverse_projection(staged,record,{},'__HELPER__')

    def test_repeated_observations_reuse_visitor_and_current_root(self):
        path='internal/t1/lib/transformer/tokenizer_internal.esk'
        staged,_=g.transform((g.ROOT/path).read_text(),path);forms=parse(staged)
        h=Host();h.helpers('cpuv1-resource-walk!')
        observed=[];errors=[];visitors=[];canonical={}
        h.env.update({'t1-private-tokenizer-registry':[],
            't1-private-entry':lambda shell:canonical.get(shell),
            'cpuv1-resource-state!':lambda *args:observed.append(args),
            'cpuv1-resource-reject!':lambda *args:errors.append(args),
            'cpuv1-resource-value-into!':lambda *args:0,
            'cpuv1-resource-seal!':lambda *args:0})
        walk_value=h.env['cpuv1-resource-walk!']
        def observe_walk(*args):
            visitors.append(args[3]);return walk_value(*args)
        h.env['cpuv1-resource-walk!']=observe_walk
        for name in ('cpuv1-resource-row-7',g.getter_name(path)):
            h.define(next(f for f in forms if f[0]=='define'
                          and isinstance(f[1],list) and f[1][0]==name))
        binding=next(f for f in forms if f[:2]==['define','cpuv1-resource-visitor-7'])
        h.env[binding[1]]=h.ev(binding[2])
        visitor=h.env[binding[1]];shell=object()
        for registry in ([],[[shell,object()]],[[shell,object()]]):
            h.env['t1-private-tokenizer-registry']=registry
            if registry:canonical[shell]=registry[0]
            h.env[g.getter_name(path)](object())
        self.assertEqual(len(observed),2);self.assertFalse(errors)
        self.assertTrue(visitors and all(v is visitor for v in visitors))
        # A copied old head fails the original canonical row guard.
        stale=[shell,object()];h.env['t1-private-tokenizer-registry']=[stale]
        h.env[g.getter_name(path)](object());self.assertTrue(errors)

    def test_inverse_rejects_reboxing_lazy_rebind_and_copied_capture(self):
        path='internal/t1/lib/transformer/tokenizer_internal.esk'
        staged,record=g.transform((g.ROOT/path).read_text(),path)
        binding='(define cpuv1-resource-visitor-7 cpuv1-resource-row-7)'
        mutations=[
            staged.replace('cpuv1-resource-visitor-7 0)','cpuv1-resource-row-7 0)',1),
            staged.replace(binding,'',1),
            staged.replace(binding,'(define (cpuv1-resource-visitor-7 packet row) (cpuv1-resource-row-7 packet row))',1),
            staged.replace('cpuv1-resource-visitor-7 0)','(begin (set! cpuv1-resource-visitor-7 cpuv1-resource-row-7) cpuv1-resource-visitor-7) 0)',1),
            staged.replace(binding,'(define cpuv1-resource-visitor-7 (let ((head t1-private-tokenizer-registry)) (lambda (packet row) head)))',1)]
        for changed in mutations:
            with self.subTest(change=changed[-500:]):
                self.assertNotEqual(changed,staged)
                with self.assertRaises(ValueError):g.reverse_projection(changed,record,{},'__HELPER__')


if __name__=='__main__':unittest.main()
