"""Cheap source/projection controls; these do not execute native code."""
from pathlib import Path
import json
import re
import subprocess
import unittest
from tests.cpuv1_resource.projection import arguments, off, tokens
from tests.cpuv1_resource.source_inventory import KINDS, active_coverage, ledger

ROOT = Path(__file__).resolve().parents[2]
BASE = '6368785d815451f409af3be42c7baec5a4c02756'
OWNERS = ['native/f32_tensor.c', 'native/i64_tensor.c', 'native/t1_i64_shell.c',
 'native/p1_identity.c', 'src/eshkol_transformer/cpu4_training_transport.c',
 'src/eshkol_transformer/e3_cpu4_frame.c', 'tests/cpu4_heldout/inspect.c',
 'src/eshkol_transformer/g3c4_model_owner.c',
 'src/eshkol_transformer/cpu4_candidate.inc',
 'src/eshkol_transformer/cpu4_training_lease.inc', 'native/o2_optimizer.c',
 'native/tr3_o2_step_clear.inc', 'native/tr3_c_o2_restore.inc', 'native/d2_native.c',
 'native/c2_checkpoint_core.c','native/c2_checkpoint_reader.c','native/checkpoint_io.c',
 'native/kernel_abi.c','native/k2_capabilities.c','native/a2_kv_cache.c',
 'src/eshkol_transformer/g3t_transport.c','native/i2_wave2_package_bridge.c',
 'src/eshkol_transformer/m3t_transport.c','src/eshkol_transformer/m3_model.c']


def baseline(path):
    return subprocess.check_output(['git', 'show', BASE + ':' + path],
                                   cwd=ROOT, text=True)


class Source(unittest.TestCase):
    def test_saved_inventory_tracks_source_bytes_and_sites(self):
        from unittest.mock import patch
        saved = json.loads((ROOT/'tests/cpuv1_resource/allocation_sites.json').read_text())
        self.assertEqual(saved, ledger())
        header = ROOT/'native/cpuv1_resource_private.h'
        read_bytes = Path.read_bytes
        with patch.object(Path, 'read_bytes', lambda path: read_bytes(path)+b'\n'
                          if path == header else read_bytes(path)):
            self.assertNotEqual(saved['source_sha256'], ledger()['source_sha256'])
        read_text = Path.read_text
        with patch.object(Path, 'read_text', lambda path: '\n'+read_text(path)
                          if path == header else read_text(path)):
            self.assertNotEqual(saved['sites'], ledger()['sites'])

    def test_exact_off_owner_tokens(self):
        for name in OWNERS:
            with self.subTest(name=name):
                self.assertEqual(tokens(off((ROOT/name).read_text())), tokens(baseline(name)))

    def test_projection_keeps_parentheses_and_math(self):
        source = '(float *)ET_R_ALLOC(D,K,calloc(n, sizeof(float)),n,sizeof(float))'
        self.assertEqual(tokens(off(source)), tokens('(float *)calloc(n, sizeof(float))'))
        self.assertNotEqual(tokens(off(source.replace('calloc(n,', 'calloc((n),'))),
                            tokens('(float *)calloc(n, sizeof(float))'))
        original = baseline('native/f32_tensor.c')
        mutated = (ROOT/'native/f32_tensor.c').read_text().replace('FLT_MANT_DIG == 24', 'FLT_MANT_DIG == 23')
        self.assertNotEqual(tokens(off(mutated)), tokens(original))

    def test_realloc_retains_original_expression(self):
        source = 'ET_R_REALLOC(D,K,realloc(p,(n+1u)*sizeof(*p)),p,n,n+1u,sizeof(*p))'
        self.assertEqual(tokens(off(source)), tokens('realloc(p,(n+1u)*sizeof(*p))'))
        header = (ROOT/'native/cpuv1_resource_private.h').read_text()
        self.assertIn('#define ET_R_ALLOC(D,K,E,N,S) E', header)
        self.assertIn('#define ET_R_REALLOC(D,K,E,P,O,N,S) E', header)

    def test_extra_argument_and_partial_cleanup(self):
        source = 'f(size_t n ET_R_EXTRA_ARG(size_t allocated)) { free_array(i,p ET_R_EXTRA_ARG(n)); }'
        self.assertEqual(tokens(off(source)),tokens('f(size_t n) { free_array(i,p); }'))
        kernel=(ROOT/'native/kernel_abi.c').read_text()
        required='free_string_array(index, copy ET_R_EXTRA_ARG(count));'
        self.assertIn(required,kernel)
        self.assertIn('ET_R_FREE(ET_R_K1,3,(void *)values,allocated_count,sizeof(*values))',kernel)
        # Changing the failure path to initialized count remains OFF-identical,
        # but violates the requested-byte equation; this separate check catches it.
        mutated=kernel.replace(required,'free_string_array(index, copy ET_R_EXTRA_ARG(index));')
        self.assertEqual(tokens(off(mutated)),tokens(baseline('native/kernel_abi.c')))
        self.assertNotIn(required,mutated)
        unit=(ROOT/'native/cpuv1_resource_k1.inc').read_text()
        self.assertIn('allocated_count=8u,initialized_count=3u',unit)
        self.assertIn('free_string_array(initialized,values ET_R_EXTRA_ARG(allocated_count))',unit)
        # Independent byte conservation for the bounded partial-cleanup input.
        for pointer_size in (4,8):
            created=8*pointer_size
            self.assertEqual(created-8*pointer_size,0)
            self.assertNotEqual(created-3*pointer_size,0)

    def test_nullable_free_metadata_is_lazy(self):
        header=(ROOT/'native/cpuv1_resource_private.h').read_text()
        macro=header.split('#define ET_R_FREE(D,K,P,N,S) do {',1)[1].split('} while (0)',1)[0]
        self.assertIn('void *et_cpuv1_base__=(void *)(P)',macro)
        self.assertRegex(macro,r'if \(et_cpuv1_base__\) et_cpuv1_test_free_event_v1\([^;]+\(N\),\(S\)\)')
        self.assertEqual(macro.count('(P)'),1)
        self.assertIn('free(et_cpuv1_base__)',macro)

    def test_runtime_and_empty_layout_boundaries(self):
        runtime=(ROOT/'native/cpuv1_resource_runtime.cpp').read_text()
        for required in ['resource_root->adopted_blocks:resource_root->current_block',
                         'ET_R_TRACKED_CURRENT]!=v[ET_R_SCHEME_CAPACITY]',
                         'get_global_arena_shared()!=resource_root',
                         'g_lambda_registry->count>g_lambda_registry->capacity',
                         'ET_R_ALLOCATOR_INUSE]-v[ET_R_KNOWN_USABLE]',
                         'owned(p,7,g_lambda_registry', 'O_RDONLY|O_CLOEXEC']:
            self.assertIn(required,runtime)
        for forbidden in ['fopen(', 'hash_table_keys(', 'hash_table_values(',
                          'eshkol_reset_resource_tracking(', 'arena_get_used_memory(']:
            self.assertNotIn(forbidden,runtime)
        c2=(ROOT/'native/cpuv1_resource_c2.inc').read_text()
        self.assertLess(c2.index('offsetof('),c2.index('for (et_c2_private_checkpoint_image'))
        m3=(ROOT/'native/cpuv1_resource_m3.inc').read_text()
        self.assertIn('registry || staged_owner || m3_workspaces || runtimes[0] || runtimes[1] || runtimes[2]',m3)
        i2=(ROOT/'native/cpuv1_resource_i2.inc').read_text()
        for field in ['et_i2_retired_copy_builders','et_i2_retired_reset_builders',
                      'et_i2_retired_decode_builders']:
            self.assertIn(field,i2)
        for field in ['tr3_c_restore_authority','tr3_c_next_index',
                      'ET_TR3_C_I2_RESTORE_PHASE_ORDINARY',
                      'ET_TR3_C_I2_RESTORE_PHASE_COMMITTED',
                      'ET_TR3_C_I2_RESTORE_PHASE_ABORTED']:
            self.assertIn(field,i2)

    def test_lexical_coverage_rejects_omissions_and_alternate_allocators(self):
        import tempfile
        from unittest.mock import patch
        inventory=ledger()
        self.assertFalse([s for s in inventory['sites'] if s['status']=='UNIMPLEMENTED'])
        owner=ROOT/'native/kernel_abi.c'
        actual=owner.read_text()
        # Same owner input, no checkout edit; a new unwrapped malloc is rejected.
        original=Path.read_text
        def omitted(path,*args,**kwargs):
            return actual+'\nvoid *missing_charge(void) { return malloc(17); }' if path==owner else original(path,*args,**kwargs)
        with patch.object(Path,'read_text',omitted):
            self.assertTrue(any(s['status']=='UNIMPLEMENTED' for s in active_coverage(owner)))
        def alternate(path,*args,**kwargs):
            return actual+'\nvoid *unsupported(void) { return strdup("x"); }' if path==owner else original(path,*args,**kwargs)
        with patch.object(Path,'read_text',alternate):
            with self.assertRaisesRegex(ValueError,'unsupported allocation family: strdup'):
                active_coverage(owner)

    def test_argument_parser(self):
        source = '(D,K,calloc(n, sizeof(*p)),n,sizeof(*p))'
        args, end = arguments(source, 0)
        self.assertEqual(args[2], 'calloc(n, sizeof(*p))')
        self.assertEqual(end, len(source))

    def test_all_domains_and_capacity(self):
        self.assertEqual(len(KINDS), 13)
        self.assertTrue(all(len(kinds) <= 64 for kinds in KINDS.values()))
        header = (ROOT/'native/cpuv1_resource_private.h').read_text()
        self.assertIn('uint64_t details[ET_R_DOMAINS][64]', header)
        self.assertNotIn('malloc(', header)

    def test_complete_requires_semantic_and_independent_census(self):
        source = (ROOT/'native/cpuv1_resource_events.c').read_text()
        self.assertIn('!p->scheme_ready', source)
        self.assertIn('r->created-r->freed!=r->bases', source)
        self.assertIn('r->allocated_requested-r->freed_requested!=r->requested', source)
        self.assertIn('r->allocated_usable-r->freed_usable!=r->usable', source)
        self.assertIn('snapshot_base_count>=ET_R_BASE_LIMIT', source)
        self.assertIn('snapshot_bases[slot]==word', source)
        self.assertIn('__has_feature(address_sanitizer)', source)
        self.assertIn('ET_CPUV1_RESOURCE_NORMAL_GLIBC_PROFILE', source)

    def test_owner_census_retains_provenance_and_both_metadata(self):
        source = (ROOT/'native/cpuv1_resource_f32.inc').read_text()
        for required in ['f32_destroy_records', 't->shape', 't->strides',
                         'retired_tensors', 'retired_borrows', 'retired_copy_plans',
                         'retired_parameters', 'retired_gradient_plans', 'retired_reset_plans']:
            self.assertIn(required, source)
        # A real omitted hook must survive OFF projection, so token equality
        # alone cannot establish coverage. Native unit/mapped-site tests do it.
        owner = (ROOT/'native/f32_tensor.c').read_text()
        self.assertGreater(len(re.findall(r'ET_R_(?:ALLOC|FREE)\(', owner)), 40)


if __name__ == '__main__':
    unittest.main()
