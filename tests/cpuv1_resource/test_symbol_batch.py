"""Source/host controls for the complete owner-scoped literal batch; no AOT proof."""
import copy
import unittest
from tests.cpu4_trainer_control.test_source_contract import parse,walk
from tests.cpuv1_resource import generate_scheme as g
from tests.cpuv1_resource.scheme_schema import RAW_NAMES,SCHEMAS
from tests.cpuv1_resource.test_review_fixes import Host,Vec


def undo_references(node,values):
    if isinstance(node,str):return copy.deepcopy(values.get(node,node))
    if node and node[0]=='quote':return node
    return [undo_references(x,values) for x in node]


class SymbolBatch(unittest.TestCase):
    def test_only_runtime_symbol_atoms_change(self):
        source='''(define existing 'already)
        (define (probe x) (list 'live (quote dead) "text 'word" '(data 'inside) '17 '#f))
        ; 'comment
        '''
        initializers,(changed,),bindings=g.prebind_observer_symbols('owner',source)
        self.assertEqual(set(bindings),{'live','dead'})
        self.assertIn('"text \'word"',changed)
        self.assertIn("'(data 'inside)",changed)
        values={form[1]:form[2] for form in parse(initializers)}
        self.assertEqual(undo_references(parse(changed),values),parse(source))
        self.assertEqual(g.observer_symbol_sites(changed),[])
        with self.assertRaises(ValueError):
            g.prebind_observer_symbols('owner',source+'\n(define '+bindings['live']+' #f)')

    def test_every_owner_has_exact_private_initializers_and_inverse(self):
        paths={r['path'] for r in g.ROOTS}|{g.typed.CANDIDATE}
        names=set();words=set();owners=0;operands=0;bindings_count=0
        for path in paths:
            text=(g.ROOT/path).read_text();staged,record=g.transform(text,path)
            symbols=record['observer_symbols'];definitions=parse(record['extra'])
            initializers=definitions[:len(symbols)]
            self.assertEqual(initializers,[['define',name,['quote',word]]
                                           for word,name in symbols.items()])
            values={f[1]:f[2] for f in initializers}
            rows=[r for r in g.ROOTS if r['path']==path]
            expected=g.typed.CANDIDATE_DEFS if path==g.typed.CANDIDATE else g.typed.definitions(path,rows)
            self.assertEqual(undo_references(definitions[len(symbols):],values),parse(expected))
            if path not in (g.P1,g.typed.CANDIDATE):
                tail=f'\n(define ({g.getter_name(path)} packet)\n'+''.join(g.root_call(r) for r in rows)+'  #t)\n'
                self.assertEqual(undo_references(parse(record['tail']),values),parse(tail))
                operands+=len(g.observer_symbol_sites(expected))+len(g.observer_symbol_sites(tail))
            self.assertFalse(g.observer_symbol_sites(record['extra']))
            self.assertFalse(g.observer_symbol_sites(record['tail']))
            for word,name in symbols.items():
                self.assertNotIn(name,text);self.assertNotIn(name,names);names.add(name)
                self.assertTrue(name.startswith('cpuv1-resource-literal-'+g.getter_name(path)[19:]+'-'))
                self.assertEqual(list(walk(parse(staged))).count(['define',name,['quote',word]]),1)
                self.assertFalse(any(n[:2]==['set!',name] for n in walk(parse(staged))))
                self.assertFalse(any(n and n[0]=='provide' and name in n for n in parse(staged)))
            if symbols:
                owners+=1;bindings_count+=len(symbols);words.update(symbols)
                word,name=next(iter(symbols.items()));literal=f"(define {name} '{word})"
                for bad in (staged.replace(literal,'',1),staged.replace(literal,literal.replace("'"+word,"'unknown"),1)):
                    with self.assertRaises(ValueError):g.reverse_projection(bad,record,{},'__HELPER__')
            g.reverse_projection(staged,record,{},'__HELPER__')
        helper=parse((g.ROOT/'tests/cpuv1_resource/helpers.esk').read_text())
        constants=[f for f in helper if f[0]=='define' and isinstance(f[1],str)
                   and f[1].startswith('cpuv1-resource-literal-')]
        self.assertEqual(len(constants),7)
        self.assertFalse(g.observer_symbol_sites((g.ROOT/'tests/cpuv1_resource/helpers.esk').read_text()))
        self.assertEqual((operands+7,bindings_count+len(constants),owners+1),(161,104,22))
        self.assertEqual(len(words|{f[2][1] for f in constants}),50)
        self.assertTrue(names.isdisjoint({f[1] for f in constants}))

    def test_shell_kind_meanings_and_unknown(self):
        h=Host();h.helpers('cpuv1-resource-p1-shell-kind')
        for word,value in [('provider',1),('module',2),('handle',3),('tree',4),
                           ('state',5),('entry',6),('state-tensor',8),('unknown',0)]:
            self.assertEqual(h.env['cpuv1-resource-p1-shell-kind'](word),value)

    def test_raw_phase_tables_keep_every_ordinal_and_unknown(self):
        covered=0
        for i,schema in SCHEMAS.items():
            expression=parse(schema['raw'])[0]
            if not isinstance(expression,list) or expression[0]!='cond':continue
            branches=expression[1:-1]
            if not all(b[0][0]=='eq?' and b[0][1][:2]==['vector-ref','row']
                       and b[0][2][0]=='quote' for b in branches):continue
            slots={int(b[0][1][2]) for b in branches};self.assertEqual(len(slots),1)
            slot=slots.pop();words=[b[0][2][1] for b in branches]
            self.assertEqual(RAW_NAMES[i],['STATIC']+words)
            source='(define (raw row) '+schema['raw']+')'
            initializers,(body,),_=g.prebind_observer_symbols('raw-test',source)
            h=Host();h.constants(initializers);h.define(parse(body)[0]);row=Vec([False]*(slot+1))
            for word,ordinal in list(zip(words,range(1,len(words)+1)))+[('unknown',15)]:
                row[slot]=word;self.assertEqual(h.env['raw'](row),ordinal)
            covered+=1
        self.assertGreaterEqual(covered,20)

    def test_actual_session_getters_load_generated_constants(self):
        for i,phases in [(59,['new','baseline','done']),(60,['new','running','done']),
                         (61,['training','handoff','adopted','released','done'])]:
            root=next(r for r in g.ROOTS if r['id']==i);path=root['path']
            staged,record=g.transform((g.ROOT/path).read_text(),path)
            h=Host();h.constants(record['extra']);h.helpers('cpuv1-resource-vector?')
            states=[];errors=[]
            def state(packet,index,broad,links,raw):
                if raw<15:states.append((index,broad,links,raw))
                else:errors.append('native-phase')
            h.env.update({'cpuv1-resource-value-into!':lambda *args:0,
                          'cpuv1-resource-seal!':lambda *args:0,
                          'cpuv1-resource-state!':state,
                          'cpuv1-resource-reject!':lambda *args:errors.append('shape-phase')})
            getter=next(f for f in parse(staged) if f[:2]==['define',[g.getter_name(path),'packet']])
            h.define(getter)
            for ordinal,phase in enumerate(phases+['unknown'],1):
                value=Vec([False,False,phase]) if i==59 else Vec([phase,False])
                if i==61:value=Vec([Vec([phase]+[False]*15)])
                h.env[root['name']]=value;states.clear();errors.clear();h.env[g.getter_name(path)](object())
                if phase=='unknown':self.assertFalse(states);self.assertTrue(errors)
                else:self.assertEqual(states,[(i,6,0,ordinal)]);self.assertFalse(errors)
            if i==61:
                h.env[root['name']]=Vec([False]);states.clear();errors.clear()
                h.env[g.getter_name(path)](object());self.assertEqual(states,[(i,6,0,0)])

    def test_protocol_operands_use_actual_symbols_and_canonical_entry(self):
        root=next(r for r in g.ROOTS if r['id']==33);path=root['path']
        staged,record=g.transform((g.ROOT/path).read_text(),path)
        h=Host();h.constants(record['extra']);h.helpers('cpuv1-resource-state-code')
        shell=object();entry=Vec([False,False,'live']);calls=[];states=[];errors=[]
        def protocol(message,arg):
            calls.append(message)
            if message=='shell':self.assertIs(arg,False);return shell
            self.assertIn(message,('check','dead'));self.assertIs(arg,entry);return True
        def state(packet,index,broad,links,raw):
            if 1<=broad<=6 and raw<15:states.append((index,broad,links,raw))
            else:errors.append('native-phase')
        h.env.update({'cpuv1-resource-g3c4-find':lambda value:entry if value is shell else False,
                      'cpuv1-resource-state!':state,
                      'cpuv1-resource-reject!':lambda *args:errors.append('identity')})
        visitor=next(f for f in parse(staged) if f[:2]==['define',['cpuv1-resource-row-33','packet','row']])
        h.define(visitor)
        for word,code in [('live',1),('dead',2),('unknown',0)]:
            entry[2]=word;calls.clear();states.clear();errors.clear()
            h.env['cpuv1-resource-row-33'](object(),protocol)
            self.assertIn('dead' if word=='dead' else 'check',calls)
            if code:self.assertEqual(states,[(33,code,2,code)]);self.assertFalse(errors)
            else:self.assertFalse(states);self.assertTrue(errors)


if __name__=='__main__':unittest.main()
