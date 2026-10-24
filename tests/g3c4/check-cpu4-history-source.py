#!/usr/bin/env python3
"""Host source/contract regression; never invokes a C preprocessor/compiler."""
from pathlib import Path
import ctypes, difflib, re, subprocess, sys
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root))
from tests.cpuv1_resource.projection import off as resource_off
owner=(root/'src/eshkol_transformer/g3c4_model_owner.c').read_text()
helper=(root/'src/eshkol_transformer/cpu4_c4_history.inc').read_text()
types=(root/'src/eshkol_transformer/cpu4_c4_history_types.h').read_text()
base=subprocess.check_output(['git','show','fd0db7270ca0212f73fc64337379d1ada32d0154:src/eshkol_transformer/g3c4_model_owner.c'],cwd=root,text=True)

def off_projection(text):
    out=[]; stack=[]; active=True
    for line in text.splitlines():
        d=line.strip()
        if re.match(r'#\s*(if|ifdef|ifndef)\b',d):
            target=d=='#ifdef ET_CPU4_C4_HISTORY_PRIVATE'
            stack.append((active,target))
            if target: active=False
            elif active: out.append(line)
        elif d=='#else':
            parent,target=stack[-1]
            if target: active=parent
            elif active: out.append(line)
        elif d=='#endif':
            parent,target=stack.pop()
            if not target and active: out.append(line)
            active=parent
        elif active: out.append(line)
    assert not stack
    text='\n'.join(out)
    text=re.sub(r'^#include "cpu4_c4_history(?:_types\.h|\.inc)"\n?', '',text,flags=re.M)
    text=re.sub(r'^#(?:define|undef) CPU4_(?:TOKEN_(?:DISPATCH|DISCOVER)|QUERY_FORWARD).*\n?', '',text,flags=re.M)
    text=text.replace('CPU4_QUERY_FORWARD(', 'et_g3c4_dispatch_forward(')
    text=text.replace('CPU4_TOKEN_DISPATCH(', 'et_g3c4_token_dispatch(').replace('CPU4_TOKEN_DISCOVER(', 'et_g3c4_token_runtime_discover(')
    return text

def tokens(t):
    t=re.sub(r'/\*.*?\*/|//[^\n]*','',t,flags=re.S)
    # Added allocation cleanup needs braces ON; one-return blocks normalize OFF.
    t=re.sub(r'\{\s*(return [^;]+;)\s*\}',r'\1',t)
    return re.findall(r'"(?:\\.|[^"\\])*"|[A-Za-z_]\w*|0[xX][\da-fA-F]+|\d+|[^\s]',t)
def physical_directives(text):
    # C translation phase2: splice BEFORE recognizing preprocessing directives.
    logical=re.sub(r'\\\r?\n','',text)
    for line in logical.splitlines():
        if re.match(r'^\s*#\s*define\b',line):
            replacement=re.sub(r'^\s*#\s*define\s+\w+(?:\([^)]*\))?','',line)
            assert not re.search(r'#\s*(?:if|ifdef|ifndef|elif|else|endif)\b',replacement), 'directive hidden by line splicing'
physical_directives(owner)
bad58f=subprocess.check_output(['git','show','58f35af2d595720981c4666ab97ae30af885a259:src/eshkol_transformer/g3c4_model_owner.c'],cwd=root,text=True)
try: physical_directives(bad58f)
except AssertionError: pass
else: raise AssertionError('physical-line control missed actual58f defect')
old=tokens(base);new=tokens(off_projection(resource_off(owner)))
if old!=new:
    difference=list(difflib.unified_diff(old,new,n=3))
    raise AssertionError('OFF source path differs: '+' '.join(difference[:140]))

class History(ctypes.Structure):
    _fields_=[('ids',ctypes.c_int64*4),('origin',ctypes.c_uint32*4),
              ('counts',ctypes.c_uint32*6),('epoch',ctypes.c_uint64)]
class Pending(ctypes.Structure):
    _fields_=[('pointers',ctypes.c_void_p*3),('epoch',ctypes.c_uint64),
              ('operation',ctypes.c_uint32),('capacity',ctypes.c_uint32)]
class Node(ctypes.Structure):
    _fields_=[('links',ctypes.c_void_p*4),('state',ctypes.c_uint32),
              ('kind',ctypes.c_uint32),('policy',ctypes.c_int64*6),
              ('rng',ctypes.c_int64*4),('history',History),
              ('parameters',ctypes.c_void_p*14),('handles',ctypes.c_void_p*14),
              ('pending',Pending)]
assert ctypes.sizeof(History)==80 and ctypes.sizeof(Pending)==40
assert Node.pending.offset==424 and ctypes.sizeof(Node)==464
assert ctypes.sizeof(History)+ctypes.sizeof(Node)==544
for flag in ['GENERATOR','ACTIVE_CALL','TOKEN_FRAME','TOKEN_FORWARD','NATIVE_PINS','PREFILL3','OUTPUT_DECODE_IDS']:
    assert f'!defined(ET_G3C4_{flag}_PRIVATE)' in types
for flag in ['ET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE','ET_I64_TENSOR_STORAGE_QUERY_PRIVATE','ET_F32_TENSOR_STORAGE_QUERY_PRIVATE']:
    assert f'!defined({flag})' in types
assert '#error ' in types

def body(source,name):
    m=list(re.finditer(r'(?m)^.*\b'+re.escape(name)+r'\([^;]*?\) \{',source))
    assert len(m)==1,(name,len(m))
    return source[m[0].start():source.index('\n}\n',m[0].end())]
def early_checks(source):
    for name,effect in [
      ('et_g3c4_private_token_frame_begin_v1','et_g3c4_private_sample_last_v1('),
      ('et_g3c4_token_frame_begin_last','memcpy(private_logits'),
      ('et_g3c4_private_prefill1_v1','CPU4_TOKEN_DISCOVER('),
      ('et_g3c4_private_prefill3_v1','CPU4_TOKEN_DISCOVER('),
      ('et_g3c4_prefill2_impl','CPU4_TOKEN_DISCOVER('),
      ('et_g3c4_private_frame_begin_v1','et_g3c4_manual_frame_allocate('),
      ('et_g3c4_p2g2_first_frame_carrier','bridge_result = et_g3c4_private_prompt_prefill_v1('),
      ('et_g3c4_private_p2g2_second_frame_carrier_v1','memcpy(last_logits')]:
        b=body(source,name)
        assert b.index('cpu4_history_preflight(')<b.index(effect),name
    assert '4u,2u)' in body(source,'et_g3c4_p2g2_first_frame_carrier')
    assert 'context->budget == 0 ? 1u : 2u' in body(source,'et_g3c4_private_prompt_prefill_v1')
    for name in ['et_g3c4_private_token_frame_publish_v1','et_g3c4_private_generation_frame_commit_v1','et_g3c4_private_p2g2_prefix_commit_v1','et_g3c4_private_p2g2_terminal_commit_v1','et_g3c4_private_frame_commit_v1']:
        b=body(source,name)
        assert b.index('cpu4_history_preflight(')<b.index('cpu4_history_staged_check(')<b.index('et_a2_kv_cache_transaction_commit_v1(')<b.index('cpu4_history_publish('),name
        tail=b[b.index('cpu4_history_publish('):]
        assert 'cpu4_history_auth(' not in tail and 'cpu4_history_preflight(' not in tail
    for name in ['et_g3c4_private_prefill1_v1','et_g3c4_prefill2_impl','et_g3c4_private_prefill3_v1']:
        b=body(source,name)
        assert b.index('cpu4_history_auth(context) != history')<b.index('cpu4_history_staged_check(')<b.index('et_a2_kv_cache_transaction_commit_v1(')<b.index('cpu4_history_publish('),name
        assert b.index('history->pending.candidate = candidate_cache')<b.index('history->pending.transaction = transaction'),name
    terminal=body(source,'et_g3c4_private_p2g2_terminal_commit_v1')
    assert terminal.index('cpu4_history_staged_check(')<terminal.index('et_i64_tensor_copy_from_v1(')
    for name in ['et_g3c4_private_sample_last_v1','et_g3c4_private_full_prefix_forward_v1']:
        b=body(source,name)
        assert 'cpu4_history_preflight(context,0u,0u,0u,0u)' in b,name
        assert 'cpu4_history_preflight(context,2u' not in b,name
    assert 'cpu4_history_publish' not in body(source,'et_g3c4_private_call_abort_v1')
    assert 'cpu4_history_publish' not in body(source,'et_g3c4_manual_a2_run')
def manual_custody_checks(source):
    b=body(source,'et_g3c4_manual_a2_run')
    begin=b.index('candidate, 1u, &count_view, &prefix, &error)')
    enroll=b.index('history->pending.transaction = prefix;',begin)
    frame=b.index('frame->a2_transaction = prefix;',enroll)
    stage=b.index('et_a2_kv_cache_transaction_stage_layer_v1(',frame)
    commit=b.index('et_a2_kv_cache_transaction_commit_v1(',stage)
    clear=b.index('history->pending.transaction = NULL;',commit)
    final=b.index('candidate, (uint64_t)count, &count_view',clear)
    assert begin<enroll<frame<stage<commit<clear<final
    assert 'cpu4_history_publish(' not in b
    c=body(source,'et_g3c4_private_frame_commit_v1')
    assert 'cpu4_history_publish(' in c and 'cpu4_history_pending_clear(' not in c
    publish=body(helper,'cpu4_history_publish')
    assert 'h->pending.operation >= 3u && h->pending.frame != NULL' in publish
    assert 'h->pending.epoch=r->commit_epoch' in publish
    retained=publish[publish.index('if (h->pending.operation >= 3u'):publish.index('} else cpu4_history_pending_clear(h);')]
    assert 'pending.frame=' not in retained and 'pending.operation=' not in retained and 'pending_clear' not in retained
    finish=body(source,'et_g3c4_private_call_finish_v1')
    assert finish.index('cpu4_history_auth(context)')<finish.index('free(context->manual_frame)')<finish.index('cpu4_history_pending_clear(history)')<finish.index('et_g3c4_active_call_drain(context)')
manual_custody_checks(resource_off(owner))
try: manual_custody_checks(bad58f)
except (AssertionError,ValueError): pass
else: raise AssertionError('manual custody control missed original58f omission')
manual_fixture=(root/'tests/g3c4/test_cpu4_history_manual_native.c').read_text()
assert 'prefix_cut_phase==1' in manual_fixture and 'prefix_cut_phase==2' in manual_fixture
assert 'prefix_custody_witnesses==witnessed+1u' in manual_fixture
assert 'h->pending.epoch=UINT64_MAX' in manual_fixture
assert 'publication_state == 3u' in helper
assert helper.index('publication_state == 3u')<helper.index('if (!h->pending.operation) return 0;')
early_checks(owner)
# Causal negative source controls: remove/move actual guard; checker must fail.
mutants=[owner.replace('cpu4_history_preflight(context,2u,1u,4u,1u)','cpu4_history_omitted(context,2u,1u,4u,1u)',1),
         owner.replace('cpu4_history_preflight(context,1u,2u,4u,2u)','cpu4_history_preflight(context,1u,2u,4u,1u)',1),
         owner.replace('cpu4_history_staged_check(history,context->token_frame_transaction','cpu4_history_omitted(history,context->token_frame_transaction',1)]
for mutant in mutants:
    try: early_checks(mutant)
    except (AssertionError,ValueError): pass
    else: raise AssertionError('ordering/custody mutation escaped guard')
assert 'if (!op) return h;' in helper
assert helper.index('if (!op) return h;')<helper.index('UINT64_MAX-epochs')
assert 'h->history.count>=cap' in helper and 'n>target_capacity' in helper
assert 'cpu4_history_staged_check' in helper
# Local test helpers must exist in each actual quoted C/.inc include topology.
# This bounded lexical control is not a C preprocessor/compiler admission.
fixture_names=['test_cpu4_history_native.c','test_cpu4_capture_native.c',
               'test_cpu4_history_manual_native.c']
def fixture_text(path,overrides):
    return overrides.get(path,path.read_text())
def c_code(text):
    return re.sub(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"', '',text,flags=re.S)
def fixture_helpers(overrides=None):
    overrides={} if overrides is None else overrides
    for name in fixture_names:
        entry=root/'tests/g3c4'/name
        seen={}
        def visit(path):
            path=path.resolve()
            if path in seen: return
            text=fixture_text(path,overrides);seen[path]=text
            for included in re.findall(r'^\s*#\s*include\s*"([^"\n]+\.(?:c|inc))"',text,re.M):
                target=(path.parent/included).resolve()
                if included == 'cpuv1_resource_c4.inc' and not target.is_file():
                    target=(root/'native'/included).resolve()
                assert target.is_relative_to(root) and target.is_file(),included
                visit(target)
        visit(entry)
        assert root/'tests/g3c4/test_prefill3.c' in seen,name
        assert root/'tests/g3c4/test_generator_native.c' not in seen,name
        definitions=set(re.findall(r'\b([A-Za-z_]\w*)\s*\([^;{}]*?\)\s*\{',
                                   '\n'.join(c_code(t) for t in seen.values()),re.S))
        calls=set(re.findall(r'\b([A-Za-z_]\w*)\s*\(',c_code(seen[entry])))
        fixed={'if','for','while','switch','sizeof','_Alignas','_Static_assert',
               'CHECK','OK','INT64_C','UINT64_C','memcpy','memset','memcmp','printf','strcmp'}
        local={n for n in calls-fixed if not n.startswith(('et_','cpu4_','__real_'))}
        missing=local-definitions
        assert not missing,(name,'unavailable fixture helpers',sorted(missing))
    history=fixture_text(root/'tests/g3c4/test_cpu4_history_native.c',overrides)
    capture=fixture_text(root/'tests/g3c4/test_cpu4_capture_native.c',overrides)
    # Seed7 reuses the genuine inherited greedy budget1 constructor.
    inherited=(root/'tests/g3c4/test_prefill3.c').read_text()
    ordinary=body(inherited,'create_generator')
    assert 'owner, 7, 0, INT64_C(0x3f800000), 256,' in ordinary
    assert 'INT64_C(0x3f800000), 1, -1)' in ordinary
    assert 'create_generator(owner)' in history
    seeded=body(capture,'capture_seed_generator')
    assert 'owner, seed, 0, INT64_C(0x3f800000), 256,' in seeded
    assert 'INT64_C(0x3f800000), 1, -1)' in seeded
    assert 'CHECK(context != NULL)' in seeded
    assert re.findall(r'capture_seed_generator\(owner,\s*(\d+)\)',capture)==['9','7','8']
fixture_helpers()
original78={root/'tests/g3c4'/name:subprocess.check_output(
    ['git','show','78aec5b0972058c3e8a2bddc56a09302e1925597:tests/g3c4/'+name],cwd=root,text=True)
    for name in fixture_names}
for name in fixture_names[:2]:
    path=root/'tests/g3c4'/name
    try: fixture_helpers({path:original78[path]})
    except AssertionError as error:
        assert 'seed_generator' in str(error)
    else: raise AssertionError(('actual78 unavailable-helper defect escaped guard',name))
capture_path=root/'tests/g3c4/test_cpu4_capture_native.c'
for before,after in [('owner, seed, 0,','owner, seed, 1,'),
                     ('INT64_C(0x3f800000), 1, -1)','INT64_C(0x3f800000), 2, -1)')]:
    mutated=capture_path.read_text().replace(before,after,1)
    try: fixture_helpers({capture_path:mutated})
    except AssertionError: pass
    else: raise AssertionError('seeded fixture policy mutation escaped guard')
print('PASS: all3 actual fixture include topologies resolve local helpers; actual78 unavailable helper rejected; exact greedy seed/policy/budget controls reject 2 mutations.')
print('PASS: physical line-splicing and manual-custody controls reject58f; OFF source tokens unchanged; effective tuple guard; LP64 80/40/464/544; early/final custody ordering and 3 causal negative controls.')
print('NOT RUN: C compiler, native layout, numerical/runtime/provider/fault gates.')
