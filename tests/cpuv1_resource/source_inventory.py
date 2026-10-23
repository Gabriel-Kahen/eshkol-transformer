"""Lexical coverage ledger only; never preprocesses or compiles project code."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
DOMAINS = {
 'native/f32_tensor.c': 'F32', 'native/p1_identity.c': 'P1',
 'src/eshkol_transformer/g3c4_model_owner.c': 'C4',
 'src/eshkol_transformer/cpu4_candidate.inc': 'C4',
 'src/eshkol_transformer/cpu4_training_lease.inc': 'C4',
 'src/eshkol_transformer/cpu4_training_transport.c': 'TRAIN',
 'src/eshkol_transformer/e3_cpu4_frame.c': 'E3',
 'tests/cpu4_heldout/inspect.c': 'E3',
 'native/o2_optimizer.c': 'O2', 'native/tr3_o2_step_clear.inc': 'O2',
 'native/tr3_c_o2_restore.inc': 'O2', 'native/i64_tensor.c': 'I64_T1',
 'native/t1_i64_shell.c': 'I64_T1',
 'src/eshkol_transformer/g3t_transport.c': 'GENERATION',
 'native/a2_kv_cache.c': 'GENERATION', 'native/d2_native.c': 'D2',
 'native/e3_d2_native.c': 'D2', 'native/c2_checkpoint_core.c': 'C2_IO',
 'native/c2_checkpoint_reader.c': 'C2_IO', 'native/checkpoint_io.c': 'C2_IO',
 'native/data_io.c': 'C2_IO', 'native/kernel_abi.c': 'K1',
}
KINDS = {
 'F32': ['tensor','borrow','copy_plan','parameter','gradient_plan','reset_plan',
         'data','shape','strides','assignments','gradient_entries',
         'prepared_gradient','reset_parameters','destroy_provenance','save_witness'],
 'P1': ['token','record','context','construction','roster'],
 'C4': ['owner','candidate','lease','child','context','rng','input','output',
        'logits','typed_result','terminal_ids','result','manual_frame'],
 'TRAIN': ['input','workspace','logits'],
 'E3': ['frame','workspace','payload','inspector_image','inspector_output'],
 'O2': ['builder','optimizer','state','state_handle','state_borrow',
        'reconstruct_builder','retire_provenance','entries','state_entries',
        'state_handles','state_clone_pointers','stages','assignments',
        'parameters','step_plan','step_stages','step_assignments','step_parameters',
        'reconstruct_entries'],
 'I64_T1': ['tensor','borrow','data','shape','strides','t1_shell'],
 'GENERATION': ['g3t_context','g3t_input','g3t_logits','g3t_output','g3t_ids',
                'g3t_lengths','g3t_cache_lengths','g3t_rng','g3t_save_hold'] +
               ['reserved'] * 7 + ['a2_cache','a2_transaction','a2_view',
                'a2_borrow','a2_keys','a2_values','a2_lengths','a2_keep_mask',
                'a2_txn_counts','a2_txn_lengths','a2_txn_keep_mask','a2_txn_staged',
                'a2_borrow_keys','a2_borrow_values'],
 'D2': ['dataset','shuffle','batch','mask'],
 'C2_IO': ['image','reader','path_parent','path_base','read_temporary','save_temporary'],
 'K1': ['runtime','capabilities','string','string_pointers','shape_ranges','dimensions'],
 'SCHEME': ['owning_scope_semantic_root_' + str(n) for n in range(64)],
 'RUNTIME': ['arena_control','block_header','block_memory','scope','mutex',
             'reflection_entries','observer_buffer'],
}
CALL = re.compile(r'\b(malloc|calloc|realloc|free|[A-Za-z_]\w*_calloc|'
                  r'e3_allocate|c4t_allocate|c2_malloc|et_g3c4_allocate)\s*\(')


def masked(text):
    # Preserve offsets/line numbers while removing comments and literals.
    return re.sub(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
                  lambda m: ''.join('\n' if c == '\n' else ' ' for c in m[0]),
                  text, flags=re.S)


def calls(path):
    text = path.read_text(); clean = masked(text)
    for m in CALL.finditer(clean):
        i = m.end(); depth = 1
        while i < len(clean) and depth:
            depth += (clean[i] == '(') - (clean[i] == ')'); i += 1
        assert depth == 0
        if clean[i:].lstrip().startswith('{'):
            continue  # allocator helper definition, not an allocation event
        yield {'line': text.count('\n', 0, m.start()) + 1,
               'callee': m[1], 'expression': text[m.start():i]}


def ledger():
    roster = json.loads((ROOT / 'native/cpu4_v1_private_inputs.json').read_text())
    files = {}; includes = {}; units = roster['translation_units'] + [
        {'source': 'tests/cpu4_heldout/inspect.c', 'defines': []},
        {'source': 'tests/cpu4_candidate_children/native_observer.c', 'defines': []}]
    def visit(path, owner):
        name = path.relative_to(ROOT).as_posix()
        if name in files: return
        text = path.read_text(); files[name] = owner
        for child in re.findall(r'^\s*#include\s+"([^"\n]+)"', text, re.M):
            choices = [path.parent / child] + [ROOT / d / child for d in roster['native_include_dirs']]
            target = next((p.resolve() for p in choices if p.is_file()), None)
            if target: visit(target, owner)
    for unit in units: visit(ROOT / unit['source'], unit['source'])
    sites = []
    for name, owner in sorted(files.items()):
        domain = DOMAINS.get(name)
        off = 'cpu4_c4_history' in name or 'cpu4_c4_capture' in name
        for call in calls(ROOT / name):
            sites.append(dict(path=name, owner=owner, domain=domain,
                              status='HISTORY_CAPTURE_OFF' if off else 'UNIMPLEMENTED',
                              **call))
    return {'schema': 'cpuv1-resource-active-site-ledger-v1',
            'base': '6368785d815451f409af3be42c7baec5a4c02756',
            'meaning': 'Lexical include superset, no preprocessor. Coverage UNSET until every active site/kind and owning census is implemented and reviewed; wrapper delegation and inactive guards must be explicit.',
            'domains': KINDS, 'translation_units': units,
            'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                              for name in sorted(files)}, 'sites': sites}


if __name__ == '__main__':
    print(json.dumps(ledger(), indent=2))
