"""Cheap source/projection controls; these do not execute native code."""
from pathlib import Path
import re
import subprocess
import unittest
from tests.cpuv1_resource.projection import arguments, off, tokens
from tests.cpuv1_resource.source_inventory import KINDS

ROOT = Path(__file__).resolve().parents[2]
BASE = '6368785d815451f409af3be42c7baec5a4c02756'
OWNERS = ['native/f32_tensor.c', 'native/i64_tensor.c', 'native/t1_i64_shell.c',
 'native/p1_identity.c', 'src/eshkol_transformer/cpu4_training_transport.c',
 'src/eshkol_transformer/e3_cpu4_frame.c', 'tests/cpu4_heldout/inspect.c',
 'src/eshkol_transformer/g3c4_model_owner.c',
 'src/eshkol_transformer/cpu4_candidate.inc',
 'src/eshkol_transformer/cpu4_training_lease.inc']


def baseline(path):
    return subprocess.check_output(['git', 'show', BASE + ':' + path],
                                   cwd=ROOT, text=True)


class Source(unittest.TestCase):
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
