#!/usr/bin/env python3
"""Host source/contract regression; never invokes a C preprocessor/compiler."""
from pathlib import Path
import ctypes, difflib, re, subprocess
root=Path(__file__).resolve().parents[2]
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
    text=re.sub(r'^#define CPU4_TOKEN_(?:DISPATCH|DISCOVER).*\n?', '',text,flags=re.M)
    text=text.replace('CPU4_TOKEN_DISPATCH(', 'et_g3c4_token_dispatch(').replace('CPU4_TOKEN_DISCOVER(', 'et_g3c4_token_runtime_discover(')
    return text

def tokens(t):
    t=re.sub(r'/\*.*?\*/|//[^\n]*','',t,flags=re.S)
    # Added allocation cleanup needs braces ON; one-return blocks normalize OFF.
    t=re.sub(r'\{\s*(return [^;]+;)\s*\}',r'\1',t)
    return re.findall(r'"(?:\\.|[^"\\])*"|[A-Za-z_]\w*|0[xX][\da-fA-F]+|\d+|[^\s]',t)
old=tokens(base);new=tokens(off_projection(owner))
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
print('PASS: OFF source tokens unchanged; effective tuple guard; LP64 80/40/464/544; early/final custody ordering and 3 causal negative controls.')
print('NOT RUN: C compiler, native layout, numerical/runtime/provider/fault gates.')
