"""Generate one noninstalled owning-scope graph; no compiler or execution.

Every edge keeps require/load flavor and resolves to one explicit literal file.
Reverse projection is exact AST/source structure, not a substitute for real AOT
canonical-head sharing or allocation-free execution proof.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from tests.cpu4_trainer_control.test_source_contract import parse
from tests.cpu4_v1_aggregate.closure import load_roster
from tests.cpuv1_resource import typed_scheme as typed

ROOT=Path(__file__).resolve().parents[2]
P1='internal/p1/lib/transformer/module.esk'
P1_SHA='e1249050c340fdd246f337a688f329462a1e1aa94720881d0cc5f9a3e8c47a94'
ROOTS=json.loads((ROOT/'tests/cpuv1_resource/scheme_roots.json').read_text())
ENTRIES=['native/cpu4_v1_private_root.esk','tests/cpu4_heldout/body.esk',
         'tests/cpu4_resume/body.esk','tests/cpu4_trained_g2/generation.esk']

@dataclass
class Node:
    start:int
    end:int
    children:list


def spans(text):
    matches=list(re.finditer(r';[^\n]*|"(?:\\.|[^"\\])*"|[()\']|[^\s()\';]+',text))
    matches=[m for m in matches if not m[0].startswith(';')]
    index=0
    def datum():
        nonlocal index
        m=matches[index];index+=1;children=[]
        if m[0]=='(':
            while matches[index][0]!=')': children.append(datum())
            end=matches[index].end();index+=1
        elif m[0]=="'":
            children=[datum()];end=children[0].end
        elif m[0]==')':raise ValueError('unexpected close')
        else:end=m.end()
        return Node(m.start(),end,children)
    answer=[]
    while index<len(matches):answer.append(datum())
    return answer


def splice(text,edits):
    for a,b,value in sorted(edits,reverse=True):text=text[:a]+value+text[b:]
    return text


def getter_name(path):
    return 'cpuv1-resource-own-'+hashlib.sha256(path.encode()).hexdigest()[:12]


def resolve(path,flavor,name):
    # String require paths are already literal; symbolic require is module syntax.
    spelling=name if flavor=='load' or '/' in name or name.endswith('.esk') else name.replace('.','/')+'.esk'
    candidates=[path.parent/spelling]+[ROOT/d/spelling for d in load_roster()['source_include_dirs']]
    target=next((p.resolve() for p in candidates if p.is_file()),None)
    if target is None:raise ValueError('unresolved edge: '+str(path)+':'+name)
    return target.relative_to(ROOT).as_posix()


def root_call(row):
    return (f"  (cpuv1-resource-value-into! packet {row['id']} {row['name']} 0)\n"
            +'  '+typed.typed_call(row)+'\n')


def p1_census(rows,lexical):
    calls=''.join(root_call(row) for row in rows)
    calls+=''.join(f'  (cpuv1-resource-value-into! packet {i} {name} 1)\n' for i,name in enumerate(lexical))
    # Existing dynamic holder reads and genuine index lookup. No materialized
    # hash keys/values, copied head or duplicated provider/native context.
    return '''(lambda (packet)
  (if (or tensor-provider-admission-active? protected-storage-comparison-active?
          construction-busy? (active-construction)
          (not (vector-ref active-record-root 2))
          (vector-ref active-record-root 3) (vector-ref e3-mode-registry-root 1))
      (cpuv1-resource-reject! packet))
  (cpuv1-resource-p1-shells packet (shell-registry-current) shell-identity-index
    native-context dead-construction-raw dead-state-raw dead-state-tensor-raw
    dead-state-entry-raw 0)
'''+calls+'''  (cpuv1-resource-p1-native-context! packet native-context)
  #t)'''


def transform(text,path):
    rows=[r for r in ROOTS if r['path']==path]
    forms=parse(text);nodes=spans(text);edits=[];extra='';tail=''
    if rows or path==typed.CANDIDATE:
        # Getter imports are only in generated test copies.
        pre='(require "__HELPER__")\n'
    else:pre=''
    for f,n in zip(forms,nodes):
        if f[0]=='provide' and rows:
            edits.append((n.end-1,n.end-1,' '+getter_name(path)))
    if path==P1:
        if hashlib.sha256(text.encode()).hexdigest()!=P1_SHA:raise ValueError('P1 immutable source mismatch')
        idx=next(i for i,f in enumerate(forms) if f[:2]==['define','p1-trusted-surface'])
        f,n=forms[idx],nodes[idx];let=n.children[2];vector=let.children[-1]
        if f[2][-1][0]!='vector' or len(f[2][-1])!=76:raise ValueError('P1 trusted slots mismatch')
        lexical=[d[1] for d in f[2][2:-1] if d[0]=='define' and isinstance(d[1],str)]
        if len(lexical)!=51:raise ValueError('P1 lexical inventory mismatch')
        name=n.children[1];edits.append((name.start,name.end,'cpuv1-p1-test-envelope'))
        original_vector=text[vector.start:vector.end]
        extra=typed.definitions(path,rows)
        edits.append((vector.start,vector.start,extra))
        edits.append((vector.start,vector.end,'(vector '+original_vector+'\n'+p1_census(rows,lexical)+')'))
        # Alias before all original wrappers: one staged published envelope.
        alias='\n(define p1-trusted-surface (vector-ref cpuv1-p1-test-envelope 0))\n'
        edits.append((n.end,n.end,alias))
        tail=f'\n(define ({getter_name(path)} packet) ((vector-ref cpuv1-p1-test-envelope 1) packet))\n'
    elif path==typed.CANDIDATE:
        idx=next(i for i,f in enumerate(forms) if f[:2]==['define','cpu4-candidate-surface'])
        f,n=forms[idx],nodes[idx];let=n.children[2];vector=let.children[-1]
        if f[2][-1][0]!='vector' or len(f[2][-1])!=7:raise ValueError('candidate six-slot surface mismatch')
        name=n.children[1];edits.append((name.start,name.end,'cpuv1-candidate-test-envelope'))
        extra=typed.CANDIDATE_DEFS
        edits.append((vector.start,vector.start,extra))
        original_vector=text[vector.start:vector.end]
        census="(lambda (packet) (cpuv1-resource-walk! packet 64 (vector-ref registry 0) cpuv1-resource-candidate-visitor 0))"
        lookup="(lambda (candidate) (cpuv1-resource-find-vector candidate (vector-ref registry 0) 0 0))"
        edits.append((vector.start,vector.end,'(vector '+original_vector+' '+census+' '+lookup+')'))
        edits.append((n.end,n.end,'\n(define cpu4-candidate-surface (vector-ref cpuv1-candidate-test-envelope 0))\n'))
        tail='\n(define (cpuv1-resource-candidate-into! packet) ((vector-ref cpuv1-candidate-test-envelope 1) packet))\n(define (cpuv1-resource-candidate-row candidate) ((vector-ref cpuv1-candidate-test-envelope 2) candidate))\n'
    elif rows:
        extra=typed.definitions(path,rows)
        tail=f'\n(define ({getter_name(path)} packet)\n'+''.join(root_call(r) for r in rows)+'  #t)\n'
    else:tail=''
    result=pre+splice(text,edits)+(extra if path not in (P1,typed.CANDIDATE) else '')+tail
    return result,{'path':path,'original_sha256':hashlib.sha256(text.encode()).hexdigest(),
                   'original':text,'getter':getter_name(path) if rows else None,'extra':extra,
                   'tail':tail,'candidate':path==typed.CANDIDATE,
                   'candidate_closures':parse(census+' '+lookup) if path==typed.CANDIDATE else []}


def reverse_projection(generated,record,rewrites,helper):
    """Inverse of exact generated insertions, then compare the immutable AST.

    No original source is substituted as the projected result.
    """
    forms=parse(generated)
    if record['getter'] or record['candidate']:
        if forms[0]!=['require',helper]:raise ValueError('missing exact helper edge')
        forms.pop(0)
        tail=parse(record['tail'])
        if forms[-len(tail):]!=tail:raise ValueError('owning getter changed')
        del forms[-len(tail):]
        if record['getter']:
            for f in forms:
                if f[0]=='provide':
                    if f[-1]!=record['getter']:raise ValueError('getter export changed')
                    f.pop()
    if record['extra'] and record['path'] not in (P1,typed.CANDIDATE):
        extra=parse(record['extra'])
        if forms[-len(extra):]!=extra:raise ValueError('typed owning definitions changed')
        del forms[-len(extra):]
    if record['path']==P1:
        idx=next(i for i,f in enumerate(forms) if f[:2]==['define','cpuv1-p1-test-envelope'])
        envelope=forms[idx]
        expected_alias=['define','p1-trusted-surface',['vector-ref','cpuv1-p1-test-envelope','0']]
        if forms[idx+1]!=expected_alias:raise ValueError('trusted alias changed')
        forms.pop(idx+1)
        vector=envelope[2][-1]
        if vector[0]!='vector' or len(vector)!=3 or vector[1][0]!='vector' or len(vector[1])!=76:
            raise ValueError('trusted envelope slot change')
        # Check the lexical visitors and census before removing insertions.
        extra=parse(record['extra'])
        if envelope[2][-len(extra)-1:-1]!=extra:raise ValueError('typed P1 definitions changed')
        del envelope[2][-len(extra)-1:-1]
        expected=p1_census([r for r in ROOTS if r['path']==P1],
            [d[1] for d in parse(record['original'])[idx][2][2:-1]
             if d[0]=='define' and isinstance(d[1],str)])
        if vector[2]!=parse(expected)[0]:raise ValueError('P1 census changed')
        envelope[1]='p1-trusted-surface';envelope[2][-1]=vector[1]
    if record['candidate']:
        idx=next(i for i,f in enumerate(forms) if f[:2]==['define','cpuv1-candidate-test-envelope'])
        envelope=forms[idx]
        if forms[idx+1]!=['define','cpu4-candidate-surface',['vector-ref','cpuv1-candidate-test-envelope','0']]:raise ValueError('candidate alias changed')
        forms.pop(idx+1);vector=envelope[2][-1]
        if vector[0]!='vector' or len(vector)!=4 or len(vector[1])!=7:raise ValueError('candidate slots changed')
        if vector[2:]!=record['candidate_closures']:raise ValueError('candidate owning closures changed')
        extra=parse(record['extra'])
        if envelope[2][-len(extra)-1:-1]!=extra:raise ValueError('candidate visitors changed')
        del envelope[2][-len(extra)-1:-1]
        envelope[1]='cpu4-candidate-surface';envelope[2][-1]=vector[1]
    for f in forms:
        if f[0] in ('load','require'):
            old=rewrites.get((f[0],f[1]))
            if old is not None:f[1]=old
    if forms!=parse(record['original']):raise ValueError('reverse projection mismatch: '+record['path'])
    return forms


def generate(destination):
    destination=Path(destination).resolve()
    if destination.exists():raise ValueError('output must be fresh')
    texts={};edges={}
    def visit(name):
        if name in texts:return
        text=(ROOT/name).read_text();texts[name]=text;edges[name]=[]
        for f,n in zip(parse(text),spans(text)):
            if f[0] in ('require','load'):
                target=resolve(ROOT/name,f[0],f[1]);edges[name].append((f,n,target));visit(target)
    for name in ENTRIES:visit(name)
    missing={r['path'] for r in ROOTS}-set(texts)
    if missing:raise ValueError('missing owning units: '+str(sorted(missing)))
    names={name:destination/'units'/name for name in texts}
    helper=str(destination/'resource_helpers.esk')
    records=[];getters=[]
    for name,text in texts.items():
        # Edges are rewritten first; inverse proves flavor and original spelling.
        rewrites={};changes=[]
        for f,node,target in edges[name]:
            argument=node.children[1]
            changes.append((argument.start,argument.end,json.dumps(str(names[target]))))
            rewrites[(f[0],str(names[target]))]=f[1]
        intermediate=splice(text,changes)
        # P1 exact hash is checked on original; keep transform/edge inverse separate.
        staged,record=transform(text,name)
        staged_nodes=spans(staged);staged_forms=parse(staged);edge_changes=[]
        for f,node in zip(staged_forms,staged_nodes):
            if f[0] in ('load','require') and f[1]!='__HELPER__':
                target=resolve(ROOT/name,f[0],f[1]);arg=node.children[1]
                edge_changes.append((arg.start,arg.end,json.dumps(str(names[target]))))
        staged=splice(staged,edge_changes).replace('"__HELPER__"',json.dumps(helper))
        reverse_projection(staged,record,rewrites,helper)
        path=names[name];path.parent.mkdir(parents=True,exist_ok=True);path.write_text(staged)
        if record['getter']:getters.append(record['getter'])
        records.append({k:v for k,v in record.items() if k not in ('original','extra','tail','candidate_closures')}|{'generated':str(path),
                        'generated_sha256':hashlib.sha256(staged.encode()).hexdigest()})
    (destination/'resource_helpers.esk').write_bytes((ROOT/'tests/cpuv1_resource/helpers.esk').read_bytes())
    root=''.join('(load '+json.dumps(str(names[n]))+')\n' for n in ENTRIES)
    root+='(define (cpuv1-resource-scheme-into! packet)\n'+''.join('  ('+g+' packet)\n' for g in getters)+'  (cpuv1-resource-candidate-into! packet)\n  (cpuv1-resource-scheme-finish! packet))\n'
    (destination/'root.esk').write_text(root)
    (destination/'tiny.esk').write_text('(load '+json.dumps(str(destination/'root.esk'))+')\n'+
        (ROOT/'tests/cpuv1_resource/tiny_genuine.esk').read_text())
    from tests.cpuv1_resource.scheme_schema import RAW_NAMES
    (destination/'source-states.json').write_text(json.dumps(RAW_NAMES,indent=2)+'\n')
    receipt={'schema':'cpuv1-resource-owning-scope-generation-v1','source_only':True,
             'p1_modules':1,'p1_runtime_slots':75,'semantic_roots':64,'p1_lexical_values':51,
             'reverse_projection':'PASS','units':records,'entry':str(destination/'root.esk')}
    (destination/'generation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt
