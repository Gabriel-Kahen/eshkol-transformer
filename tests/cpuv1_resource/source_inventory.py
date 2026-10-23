"""Lexical coverage ledger only; never preprocesses or compiles project code."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
DOMAINS = {
 'native/i2_wave2_package_bridge.c':'F32',
 'src/eshkol_transformer/m3t_transport.c':'TRAIN',
 'src/eshkol_transformer/m3_model.c':'TRAIN',
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
         'prepared_gradient','reset_parameters','destroy_provenance','save_witness',
         'i2_copy_builder','i2_reset_builder','i2_decode_builder','i2_decode_shape',
         'i2_decode_payload','i2_copy_assignments','i2_reset_parameters','i2_transient_words'],
 'P1': ['token','record','context','construction','roster'],
 'C4': ['owner','candidate','lease','child','context','rng','input','output',
        'logits','typed_result','terminal_ids','result','manual_frame'],
 'TRAIN': ['input','workspace','logits','m3t_initializer','m3t_owner','m3t_input',
           'm3t_logits','m3t_workspace','m3_workspace','m3_graph','m3_logits'],
 'E3': ['frame','workspace','payload','inspector_image','inspector_output'],
 'O2': ['builder','optimizer','state','state_handle','state_borrow',
        'reconstruct_builder','retire_provenance','entries','state_entries',
        'state_handles','state_clone_pointers','stages','assignments',
        'parameters','step_plan','step_stages','step_assignments','step_parameters',
        'reconstruct_entries','restore_plan','reconstruct_clone_pointers','reconstruct_entry_set'],
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
             'reflection_entries','observer_buffer','reflection_registry'],
}
CALL = re.compile(r'\b(malloc|calloc|realloc|free|[A-Za-z_]\w*_(?:calloc|malloc)|'
                  r'e3_allocate|c4t_allocate|c2_malloc|et_g3c4_allocate|allocate)\s*\(')


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
        yield {'offset': m.start(), 'end': i, 'line': text.count('\n', 0, m.start()) + 1,
               'callee': m[1], 'expression': text[m.start():i]}


UNSUPPORTED = re.compile(r'\b(strdup|strndup|aligned_alloc|posix_memalign|valloc|pvalloc|memalign|asprintf|vasprintf|getline|getdelim|realpath|fopen|fdopen|opendir|scandir|mmap|munmap)\s*\(')
DELEGATES = {
 'native/f32_tensor.c': {'calloc(count, size)'},
 'native/a2_kv_cache.c': {'calloc(count, size)'},
 'native/i2_wave2_package_bridge.c': {'calloc(count, size)','malloc(size)'},
 'native/o2_optimizer.c': {'calloc(count, size)'},
 'native/tr3_o2_step_clear.inc': {'calloc(count, size)'},
 'native/c2_checkpoint_core.c': {'malloc(bytes)'},
 'native/i64_tensor.c': {'calloc(count, size)'},
 'native/checkpoint_io.c': {'malloc(length)'},
 'src/eshkol_transformer/g3c4_model_owner.c': {'calloc(1u, size)'},
 'src/eshkol_transformer/cpu4_training_transport.c': {'calloc(1,bytes)'},
 'src/eshkol_transformer/e3_cpu4_frame.c': {'calloc(1, bytes)'},
 'src/eshkol_transformer/m3t_transport.c': {'calloc(1,n)'},
}

def active_coverage(path):
    from tests.cpuv1_resource.projection import arguments, tokens
    text=path.read_text();clean=masked(text)
    unsupported=UNSUPPORTED.search(clean)
    if unsupported: raise ValueError('unsupported allocation family: '+unsupported[1])
    wrapped=[]
    for m in re.finditer(r'\bET_R_(ALLOC|REALLOC)\s*\(',clean):
        args,end=arguments(text,m.end()-1)
        wrapped.append((m.start(),end,args[0],args[1]))
    rows=[]
    relative=path.relative_to(ROOT).as_posix()
    for row in calls(path):
        outer=next((x for x in wrapped if x[0]<row['offset']<x[1]),None)
        if outer:
            row.update(status='WRAPPED',domain=outer[2],kind=outer[3])
        elif any(tokens(row['expression'])==tokens(e) for e in DELEGATES.get(relative,set())):
            row.update(status='DELEGATED_HELPER_REAL_ALLOCATOR')
        elif relative=='src/eshkol_transformer/g3c4_model_owner.c' and row['expression']=='free(history)':
            row.update(status='HISTORY_CAPTURE_OFF')
        elif 'cpu4_c4_history' in relative or 'cpu4_c4_capture' in relative:
            row.update(status='HISTORY_CAPTURE_OFF')
        elif 'cpuv1_resource_' in relative:
            row.update(status='OBSERVER_INTERNAL_REVIEW_REQUIRED')
        else: row.update(status='UNIMPLEMENTED')
        rows.append(row)
    return rows

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
        for call in active_coverage(ROOT / name):
            sites.append(dict({'path':name,'owner':owner,'domain':domain},**call))
    return {'schema': 'cpuv1-resource-active-site-ledger-v1',
            'base': '6368785d815451f409af3be42c7baec5a4c02756',
            'meaning': 'Lexical include superset, no preprocessor. Coverage UNSET until every active site/kind and owning census is implemented and reviewed; wrapper delegation and inactive guards must be explicit.',
            'domains': KINDS, 'translation_units': units,
            'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                              for name in sorted(files)}, 'sites': sites}


if __name__ == '__main__':
    print(json.dumps(ledger(), indent=2))
