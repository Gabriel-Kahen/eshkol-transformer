/* Ownership/transaction fixtures only: readiness and finite slot bytes are
 * injected below. No C model schedule, numerical oracle or parity claim. */
#define ET_M3_TESTING 1
#define ET_M3T_TESTING 1
#include "../../src/eshkol_transformer/m3_model.c"
#include <stdio.h>
_Static_assert(sizeof(m3_graph) == 448 && sizeof(m3_logits) == 32 &&
  sizeof(m3_workspace) == 32 && sizeof(owner) == 272 && sizeof(workspace) == 760 &&
  sizeof(initializer) == 48, "x86-64 native retention layout");

static size_t checks;
#define CHECK(x) do { ++checks; if (!(x)) { \
  fprintf(stderr, "FAIL line %d: %s [domain=%lld category=%lld code=%lld]\n", \
    __LINE__, #x, (long long)error_domain, (long long)error_category, \
    (long long)error_code); exit(1); } } while (0)
#define OK(x) CHECK((x) == 0)
static et_f32_tensor_error fe;
static et_i64_tensor_error ie;
static unsigned char identities[2][14];

typedef struct snapshot {
  uint32_t value[1184], gradient[1184];
  et_f32_gradient_metadata_v1 metadata[14];
} snapshot;
typedef struct live_snapshot {
  et_f32_test_live_counts_v1 f32;
  int64_t i1[5];
} live_snapshot;
static live_snapshot live(void) {
  live_snapshot s = {.f32 = {.struct_size = sizeof(s.f32)}};
  et_f32_test_live_counts_snapshot_v1(&s.f32);
  for (int n = 0; n < 5; ++n) s.i1[n] = et_m3_test_i64_counts_v1(n);
  return s;
}
static void live_equal(live_snapshot a, live_snapshot b) {
  CHECK(a.f32.tensors == b.f32.tensors && a.f32.parameters == b.f32.parameters);
  CHECK(a.f32.borrows == b.f32.borrows && a.f32.copy_plans == b.f32.copy_plans);
  CHECK(a.f32.gradient_plans == b.f32.gradient_plans && a.f32.reset_plans == b.f32.reset_plans);
  CHECK(a.f32.owned_clones == b.f32.owned_clones);
  CHECK(memcmp(a.i1, b.i1, sizeof(a.i1)) == 0);
}
static et_f32_test_retired_counts_v1 retired(void) {
  et_f32_test_retired_counts_v1 r = {.struct_size = sizeof(r)};
  et_f32_test_retired_counts_snapshot_v1(&r); return r;
}
static void save(owner *o, snapshot *s) {
  size_t offset = 0;
  memset(s, 0, sizeof(*s));
  for (size_t i = 0; i < 14; ++i) {
    size_t n = 0; et_f32_tensor *v = value(o, i);
    OK(et_f32_tensor_element_count_v1(v, &n, &fe));
    OK(et_f32_tensor_copy_bits_to_v1(v, s->value + offset, n, &fe));
    OK(et_f32_tensor_copy_bits_to_v1(et_m3t_test_parameter_gradient(o->p[i]), s->gradient + offset, n, &fe));
    s->metadata[i].struct_size = sizeof(s->metadata[i]);
    OK(et_f32_parameter_gradient_metadata_v1(o->p[i], &s->metadata[i], &fe));
    offset += n;
  }
  CHECK(offset == 1184);
}
static void unchanged(owner *o, const snapshot *before) {
  snapshot after; save(o, &after); CHECK(memcmp(before, &after, sizeof(after)) == 0);
}
static void check_metadata(owner *o, uint32_t state, uint64_t count, uint32_t weight, int zero) {
  for (size_t i = 0; i < 14; ++i) {
    et_f32_gradient_metadata_v1 m = {.struct_size = sizeof(m)};
    OK(et_f32_parameter_gradient_metadata_v1(o->p[i], &m, &fe));
    CHECK(m.state == state && m.contribution_count == count && m.normalization_weight_bits == weight);
    if (zero) {
      uint32_t bits[1024]; size_t n = 0;
      et_f32_tensor *g = et_m3t_test_parameter_gradient(o->p[i]);
      OK(et_f32_tensor_element_count_v1(g, &n, &fe));
      OK(et_f32_tensor_copy_bits_to_v1(g, bits, n, &fe));
      for (size_t j = 0; j < n; ++j) CHECK(bits[j] == 0);
    }
  }
}
static void zero_grad(owner *o) {
  et_f32_gradient_reset_plan *p = NULL;
  OK(et_f32_gradient_reset_plan_prepare_v1(14, o->p, &p, &fe));
  et_f32_tensor_test_fail_alloc_after_v1(0);
  OK(et_f32_gradient_reset_plan_commit_v1(p, &fe));
  OK(et_f32_gradient_reset_plan_release_v1(&p, &fe));
  et_f32_tensor_test_reset_allocator_v1();
}
static owner *new_owner(int row) {
  initializer *s = et_m3t_private_initializer_create_v1(1729 + row); CHECK(s);
  OK(et_m3t_private_initializer_seal_v1(s));
  owner *o = et_m3t_private_owner_create_v1(s); CHECK(o);
  for (int i = 0; i < 14; ++i) OK(et_m3t_private_owner_bind_v1(o, i, &identities[row][i]));
  for (int i = 0; i < 14; ++i) OK(et_m3t_private_owner_initialize_v1(o, i));
  CHECK(et_m3t_private_owner_successor_v1(o));
  OK(et_m3t_private_owner_seal_v1(o));
  OK(et_m3t_private_initializer_seal_v1(o->successor));
  return o;
}
static void fill(et_f32_tensor *t, uint32_t bit) {
  uint32_t bits[1024]; size_t count = 0;
  OK(et_f32_tensor_element_count_v1(t, &count, &fe)); CHECK(count <= 1024);
  for (size_t i = 0; i < count; ++i) bits[i] = bit;
  OK(et_f32_tensor_copy_bits_from_v1(t, bits, count, &fe));
}
static void forward_fixture(workspace *w, input *in, uint32_t tag) {
  OK(et_m3t_private_workspace_begin_v1(w, in, 1));
  fill(w->slots[Z], tag); w->ready[Z] = 1;
}
static m3_graph *graph_fixture(workspace *w, input *in, uint32_t tag) {
  forward_fixture(w, in, tag);
  m3_graph *g = et_m3_private_graph_capture_v1(w); CHECK(g);
  OK(et_m3_private_workspace_reset_v1(w)); return g;
}
static void reverse_fixture(workspace *w, m3_graph *g, logits *seed, uint32_t bit) {
  OK(et_m3_private_graph_restore_v1(w, g));
  /* Transport-only stand-ins, deliberately not a complete numerical schedule. */
  w->ready[Z] = 1;
  OK(et_m3t_private_workspace_vjp_begin_v1(w, seed));
  for (size_t i = 0; i < 14; ++i) { fill(w->slots[GKEY + i], bit); w->ready[GKEY + i] = 1; }
}
static void leaf_and_workspace(owner *o, input *in) {
  int foreign = 0;
  CHECK(et_m3_private_i64_unborrowed_v1((const void *)(uintptr_t)1, &ie));
  CHECK(et_m3_private_workspace_reset_v1(&foreign));
  for (size_t n = 0; n < 2; ++n) {
    live_snapshot before = live(); et_m3t_test_fail_alloc_after(n);
    CHECK(!et_m3t_private_workspace_create_v1(o)); live_equal(before, live());
    et_m3t_test_fail_alloc_after(SIZE_MAX);
  }
  workspace *old = et_m3t_private_workspace_create_v1(o); CHECK(old);
  OK(et_m3t_private_workspace_begin_v1(old, in, 1));
  OK(et_m3t_private_workspace_reset_v1(old));
  CHECK(et_m3_private_workspace_reset_v1(old));
  OK(et_m3t_private_workspace_release_v1(old));
  workspace *w = et_m3t_private_workspace_create_v1(o); CHECK(w);
  et_i64_tensor *ids[] = {w->ids, w->positions};
  for (size_t n = 0; n < 2; ++n) {
    et_i64_tensor_borrow *b = NULL;
    OK(et_i64_tensor_borrow_begin_v1(ids[n], &b, &ie));
    CHECK(et_m3_private_workspace_reset_v1(w));
    CHECK(!m3_find_workspace(w)->owned);
    OK(et_i64_tensor_borrow_end_v1(&b, &ie));
    int64_t saved[2], after[2];
    OK(et_i64_tensor_copy_to_v1(ids[n], saved, 2, &ie));
    CHECK(et_m3_private_i64_unborrowed_v1((const void *)(uintptr_t)1,
      (et_i64_tensor_error *)(void *)et_i64_tensor_test_data_storage_v1(ids[n])));
    OK(et_i64_tensor_copy_to_v1(ids[n], after, 2, &ie));
    CHECK(!memcmp(saved, after, sizeof(saved)));
  }
  et_i64_tensor_test_fail_alloc_after_v1(0); et_f32_tensor_test_fail_alloc_after_v1(0);
  OK(et_m3_private_workspace_reset_v1(w));
  OK(et_m3_private_workspace_reset_v1(w));
  OK(et_m3t_private_workspace_release_v1(w));
  OK(et_m3t_private_workspace_release_v1(w));
  CHECK(et_m3_private_workspace_reset_v1(w));
  et_i64_tensor_test_reset_allocator_v1(); et_f32_tensor_test_reset_allocator_v1();
}
static void capture_failures(workspace *w, input *in) {
  for (size_t n = 0; n < 60; ++n) {
    forward_fixture(w, in, 0x3f800000u); live_snapshot before = live();
    et_f32_tensor_test_fail_alloc_after_v1(n);
    CHECK(!et_m3_private_graph_capture_v1(w));
    live_equal(before, live());
    et_i64_tensor_test_fail_alloc_after_v1(0);
    OK(et_m3_private_workspace_reset_v1(w));
    et_i64_tensor_test_reset_allocator_v1(); et_f32_tensor_test_reset_allocator_v1();
  }
  for (size_t n = 0; n < 4; ++n) {
    forward_fixture(w, in, 0x3f800000u); live_snapshot before = live();
    et_i64_tensor_test_fail_alloc_after_v1(n);
    CHECK(!et_m3_private_graph_capture_v1(w)); live_equal(before, live());
    OK(et_m3_private_workspace_reset_v1(w)); et_i64_tensor_test_reset_allocator_v1();
  }
  forward_fixture(w, in, 0x3f800000u); live_snapshot before = live();
  et_m3t_test_fail_alloc_after(0);
  CHECK(!et_m3_private_graph_capture_v1(w)); live_equal(before, live());
  OK(et_m3_private_workspace_reset_v1(w)); et_m3t_test_fail_alloc_after(SIZE_MAX);
}
static void ownership(workspace *w, input *in, logits *seed, owner *other) {
  live_snapshot base = live();
  m3_graph *g = graph_fixture(w, in, 0x3f800000u);
  live_snapshot before_record = live(); et_m3t_test_fail_alloc_after(0);
  CHECK(!et_m3_private_model_logits_create_v1(g)); CHECK(g->leases == 1);
  live_equal(before_record, live()); et_m3t_test_fail_alloc_after(SIZE_MAX);
  for (size_t n = 0; n < 4; ++n) {
    live_snapshot before = live(); et_f32_tensor_test_fail_alloc_after_v1(n);
    CHECK(!et_m3_private_model_logits_create_v1(g)); CHECK(g->leases == 1);
    live_equal(before, live()); et_f32_tensor_test_reset_allocator_v1();
  }
  m3_logits *a = et_m3_private_model_logits_create_v1(g), *b = et_m3_private_model_logits_create_v1(g);
  CHECK(a && b && g->leases == 3);
  struct { int64_t size; uint32_t bits[512]; } bytes = {2048, {0}};
  OK(et_m3_private_model_logits_copy_to_v1(a, &bytes, 2048));
  for (size_t n = 0; n < 512; ++n) CHECK(bytes.bits[n] == 0x3f800000u);
  m3_graph *second = graph_fixture(w, in, 0x40000000u);
  OK(et_m3_private_model_logits_copy_to_v1(a, &bytes, 2048));
  for (size_t n = 0; n < 512; ++n) CHECK(bytes.bits[n] == 0x3f800000u);
  OK(et_m3_private_graph_release_v1(g)); OK(et_m3_private_graph_release_v1(g)); CHECK(g->leases == 2);
  reverse_fixture(w, g, seed, 0);
  CHECK(et_m3_private_workspace_contribute_v1(w, second, 1, 0x3f800000, 0));
  CHECK(et_m3_private_model_logits_release_v1(a)); CHECK(g->leases == 2);
  OK(et_m3_private_graph_release_v1(g)); /* Root already released: no second drop. */
  OK(et_m3_private_workspace_reset_v1(w));
  workspace *foreign = et_m3t_private_workspace_create_v1(other); CHECK(foreign);
  OK(et_m3_private_workspace_reset_v1(foreign));
  CHECK(et_m3_private_graph_restore_v1(foreign, g));
  OK(et_m3t_private_workspace_release_v1(foreign));
  for (size_t n = 0; n < 15; ++n) {
    et_f32_tensor_borrow *borrow = NULL;
    et_f32_tensor *t = n < 14 ? g->saved[n] : g->logits;
    OK(et_f32_tensor_borrow_begin_v1(t, &borrow, &fe));
    CHECK(et_m3_private_model_logits_release_v1(a)); CHECK(g->leases == 2);
    OK(et_f32_tensor_borrow_end_v1(&borrow, &fe));
  }
  et_i64_tensor_borrow *ib = NULL;
  OK(et_i64_tensor_borrow_begin_v1(g->ids, &ib, &ie));
  CHECK(et_m3_private_model_logits_release_v1(a));
  OK(et_i64_tensor_borrow_end_v1(&ib, &ie));
  et_f32_tensor_test_fail_alloc_after_v1(0); et_i64_tensor_test_fail_alloc_after_v1(0);
  OK(et_m3_private_model_logits_release_v1(a)); OK(et_m3_private_model_logits_release_v1(a));
  OK(et_m3_private_model_logits_release_v1(b)); CHECK(g->r.state == -1);
  CHECK(et_m3_private_model_logits_copy_to_v1(b, &bytes, 2048));
  CHECK(et_m3_private_graph_restore_v1(w, g)); CHECK(et_m3_private_graph_check_primals_v1(g, 1));
  CHECK(et_m3_private_model_logits_release_v1(second));
  OK(et_m3_private_graph_release_v1(second));
  et_f32_tensor_test_reset_allocator_v1(); et_i64_tensor_test_reset_allocator_v1();
  live_equal(base, live());
}
static void transaction(workspace *w, input *in, logits *seed, owner *o) {
  m3_graph *g = graph_fixture(w, in, 0x3f800000u);
  snapshot before;
  for (size_t n = 0; n < 16; ++n) {
    reverse_fixture(w, g, seed, 0); save(o, &before); live_snapshot base = live();
    et_f32_tensor_test_fail_alloc_after_v1(n); et_i64_tensor_test_fail_alloc_after_v1(0);
    CHECK(et_m3_private_workspace_contribute_v1(w, g, 1, 0x3f800000, 0));
    CHECK(error_code == ET_F32_TENSOR_CODE_ALLOCATION_FAILED);
    unchanged(o, &before); live_equal(base, live());
    OK(et_m3_private_workspace_reset_v1(w)); CHECK(!g->active && !o->active);
    et_f32_tensor_test_reset_allocator_v1(); et_i64_tensor_test_reset_allocator_v1();
  }
  reverse_fixture(w, g, seed, 0);
  et_f32_tensor_test_fail_alloc_after_v1(16); et_i64_tensor_test_fail_alloc_after_v1(0);
  OK(et_m3_private_workspace_contribute_v1(w, g, 1, 0x3f800000, 0));
  CHECK(!g->active && !o->active && w->r.state == 0);
  et_f32_tensor *fail = NULL; OK(et_f32_tensor_element_count_v1(w->epsilon, &(size_t){0}, &fe));
  CHECK(et_f32_tensor_create_v1(0, NULL, &fail, &fe)); CHECK(!fail);
  et_f32_tensor_test_reset_allocator_v1(); et_i64_tensor_test_reset_allocator_v1();
  check_metadata(o, ET_F32_GRADIENT_PRESENT, 1, 0x3f800000u, 1);
  reverse_fixture(w, g, seed, 0x3f800000u);
  OK(et_m3_private_workspace_contribute_v1(w, g, 1, 0x3f000000, 1));
  check_metadata(o, ET_F32_GRADIENT_PRESENT, 2, 0x3fc00000u, 0);
  /* Repeat every prepare allocation failure with existing nonzero numerators. */
  for (size_t n = 0; n < 16; ++n) {
    reverse_fixture(w, g, seed, 0x3f800000u); save(o, &before);
    live_snapshot base = live(); et_f32_tensor_test_fail_alloc_after_v1(n);
    CHECK(et_m3_private_workspace_contribute_v1(w, g, 1, 0x3f800000, 2));
    CHECK(error_code == ET_F32_TENSOR_CODE_ALLOCATION_FAILED);
    unchanged(o, &before); live_equal(base, live());
    OK(et_m3_private_workspace_reset_v1(w)); et_f32_tensor_test_reset_allocator_v1();
  }
  reverse_fixture(w, g, seed, 0x3f800000u);
  /* Last-entry finite+finite overflow must not update earlier destinations. */
  fill(et_m3t_test_parameter_gradient(o->p[13]), 0x7f7fffffu);
  fill(w->slots[GPOS], 0x7f7fffffu); save(o, &before);
  CHECK(et_m3_private_workspace_contribute_v1(w, g, 1, 0x3f800000, 2)); unchanged(o, &before);
  fill(et_m3t_test_parameter_gradient(o->p[13]), 0x3f800000u); fill(w->slots[GPOS], 0x3f800000u);
  for (size_t n = 0; n < 14; ++n)
    et_f32_parameter_test_set_metadata_v1(o->p[n], ET_F32_GRADIENT_PRESENT, 2, 0x7f7fffffu);
  save(o, &before);
  CHECK(et_m3_private_workspace_contribute_v1(w, g, 1, 0x7f7fffff, 2)); unchanged(o, &before);
  for (size_t n = 0; n < 14; ++n)
    et_f32_parameter_test_set_metadata_v1(o->p[n], ET_F32_GRADIENT_PRESENT, INT64_MAX, 0x3fc00000u);
  save(o, &before);
  CHECK(et_m3_private_workspace_contribute_v1(w, g, 1, 0x3f800000, INT64_MAX)); unchanged(o, &before);
  for (size_t n = 0; n < 14; ++n)
    et_f32_parameter_test_set_metadata_v1(o->p[n], ET_F32_GRADIENT_PRESENT, 2, 0x3fc00000u);
  et_f32_tensor_borrow *gradient_borrow = NULL;
  OK(et_f32_parameter_gradient_borrow_begin_v1(o->p[13], &gradient_borrow, &fe));
  save(o, &before); CHECK(et_m3_private_workspace_contribute_v1(w, g, 1, 0x3f800000, 2)); unchanged(o, &before);
  OK(et_f32_tensor_borrow_end_v1(&gradient_borrow, &fe));
  OK(et_m3_private_workspace_reset_v1(w));
  save(o, &before);
  reverse_fixture(w, g, seed, 0);
  OK(et_m3_private_workspace_contribute_v1(w, g, 1, 0x40000000, 2));
  check_metadata(o, ET_F32_GRADIENT_PRESENT, 3, 0x40600000u, 0);
  snapshot after; save(o, &after); CHECK(!memcmp(before.gradient, after.gradient, sizeof(before.gradient)));
  reverse_fixture(w, g, seed, 0); save(o, &before);
  CHECK(et_m3_private_workspace_contribute_v1(w, g, 1, 0x3f800000, 2)); unchanged(o, &before);
  CHECK(et_m3_private_workspace_contribute_v1(w, g, 0, 0x3f800000, 3)); unchanged(o, &before);
  const int64_t weights[] = {0, 0xbf800000, 0x7f800000, 0x7fc00000, INT64_C(0x100000000), -1};
  for (size_t n = 0; n < sizeof(weights) / sizeof(*weights); ++n) {
    CHECK(et_m3_private_workspace_contribute_v1(w, g, 1, weights[n], 3)); unchanged(o, &before);
  }
  fill(w->slots[GPOS], 0x7fc00000u);
  CHECK(et_m3_private_workspace_contribute_v1(w, g, 1, 0x3f800000, 3)); unchanged(o, &before);
  fill(w->slots[GPOS], 0);
  et_f32_parameter_test_set_metadata_v1(o->p[13], ET_F32_GRADIENT_PRESENT, 2, 0x40600000u);
  save(o, &before); CHECK(et_m3_private_workspace_contribute_v1(w, g, 1, 0x3f800000, 3)); unchanged(o, &before);
  et_f32_parameter_test_set_metadata_v1(o->p[13], ET_F32_GRADIENT_PRESENT, 3, 0x40600000u);
  OK(et_m3_private_workspace_reset_v1(w)); zero_grad(o);
  check_metadata(o, ET_F32_GRADIENT_ABSENT, 0, 0, 1);
  /* Both kinds of competing plan pins reject; ending them permits same-frame retry. */
  for (int kind = 0; kind < 2; ++kind) {
    reverse_fixture(w, g, seed, 0); save(o, &before);
    et_f32_gradient_plan *gp = NULL; et_f32_gradient_reset_plan *rp = NULL;
    et_f32_gradient_contribution_v1 rows[14];
    for (size_t n = 0; n < 14; ++n) rows[n] = (et_f32_gradient_contribution_v1){
      sizeof(rows[n]), o->p[n], w->slots[GKEY + n], 0};
    if (kind) OK(et_f32_gradient_plan_prepare_v1(14, rows, 0x3f800000u, &gp, &fe));
    else OK(et_f32_gradient_reset_plan_prepare_v1(14, o->p, &rp, &fe));
    CHECK(et_m3_private_workspace_contribute_v1(w, g, 1, 0x3f800000, 0)); unchanged(o, &before);
    if (kind) OK(et_f32_gradient_plan_release_v1(&gp, &fe));
    else OK(et_f32_gradient_reset_plan_release_v1(&rp, &fe));
    OK(et_m3_private_workspace_contribute_v1(w, g, 1, 0x3f800000, 0)); zero_grad(o);
  }
  /* Every canonical live value mutation is rejected; exact restoration retries. */
  for (size_t n = 0; n < 14; ++n) {
    reverse_fixture(w, g, seed, 0);
    uint32_t bits[1024], changed[1024]; size_t count = 0; et_f32_tensor *v = value(o, n);
    OK(et_f32_tensor_element_count_v1(v, &count, &fe)); OK(et_f32_tensor_copy_bits_to_v1(v, bits, count, &fe));
    memcpy(changed, bits, count * 4); changed[0] ^= UINT32_C(0x80000000);
    OK(et_m3_private_graph_check_primals_v1(g, 1));
    OK(et_f32_tensor_copy_bits_from_v1(v, changed, count, &fe)); save(o, &before);
    CHECK(et_m3_private_workspace_contribute_v1(w, g, 1, 0x3f800000, 0)); unchanged(o, &before);
    OK(et_f32_tensor_copy_bits_from_v1(v, bits, count, &fe));
    OK(et_m3_private_workspace_contribute_v1(w, g, 1, 0x3f800000, 0)); zero_grad(o);
  }
  /* Late live parameter borrow and each I1 workspace hold reject atomically. */
  reverse_fixture(w, g, seed, 0); save(o, &before);
  et_f32_tensor_borrow *borrow = NULL;
  OK(et_f32_parameter_value_borrow_begin_v1(o->p[13], &borrow, &fe));
  CHECK(et_m3_private_workspace_contribute_v1(w, g, 1, 0x3f800000, 0)); unchanged(o, &before);
  OK(et_f32_tensor_borrow_end_v1(&borrow, &fe));
  et_i64_tensor *ids[] = {w->ids, w->positions};
  for (size_t n = 0; n < 2; ++n) {
    et_i64_tensor_borrow *b = NULL; OK(et_i64_tensor_borrow_begin_v1(ids[n], &b, &ie));
    CHECK(et_m3_private_workspace_reset_v1(w));
    CHECK(et_m3_private_workspace_contribute_v1(w, g, 1, 0x3f800000, 0)); unchanged(o, &before);
    CHECK(g->active == w && o->active == w);
    OK(et_i64_tensor_borrow_end_v1(&b, &ie));
  }
  OK(et_m3_private_workspace_reset_v1(w)); OK(et_m3_private_graph_release_v1(g));
}
static void trajectory(workspace *w, input *in, logits *seed, owner *o, size_t limit) {
  live_snapshot base = live(); et_f32_test_retired_counts_v1 start = retired();
  for (size_t i = 1; i <= limit; ++i) {
    m3_graph *g = graph_fixture(w, in, 0x3f800000u);
    m3_logits *l = et_m3_private_model_logits_create_v1(g); CHECK(l);
    OK(et_m3_private_graph_release_v1(g));
    reverse_fixture(w, g, seed, 0);
    OK(et_m3_private_workspace_contribute_v1(w, g, 1, 0x3f800000, 0));
    zero_grad(o); OK(et_m3_private_model_logits_release_v1(l));
    live_equal(base, live());
    if (i == 10 || i == 100 || i == 1000) {
      et_f32_test_retired_counts_v1 now = retired();
      CHECK(now.tensors - start.tensors == 16 * i);
      CHECK(now.gradient_plans - start.gradient_plans == i);
      CHECK(now.reset_plans - start.reset_plans == i);
      CHECK(now.retained_control_bytes - start.retained_control_bytes == 1496 * i);
      printf("M3 cycles=%zu i2-retired-delta=%zu expected=%zu\n", i,
        now.retained_control_bytes - start.retained_control_bytes, 1496 * i);
      OK(et_m3_test_report_counts_v1());
    }
  }
}
int main(int argc, char **argv) {
  CHECK(argc == 1 || (argc == 2 && !strcmp(argv[1], "100")));
  size_t limit = argc == 2 ? 100 : 1000;
  owner *o = new_owner(0), *other = new_owner(1); live_snapshot model_base = live();
  input *in = et_m3t_private_input_create_v1(); logits *seed = et_m3t_private_logits_create_v1();
  CHECK(in && seed); OK(et_m3t_private_input_copy_v1(in, 65, 66));
  leaf_and_workspace(o, in);
  workspace *w = et_m3t_private_workspace_create_v1(o); CHECK(w);
  OK(et_m3_private_workspace_reset_v1(w));
  capture_failures(w, in); ownership(w, in, seed, other); transaction(w, in, seed, o);
  trajectory(w, in, seed, o, limit);
  OK(et_m3t_private_workspace_release_v1(w));
  OK(et_m3t_private_input_release_v1(in)); OK(et_m3t_private_logits_release_v1(seed));
  live_equal(model_base, live());
  printf("M3 NATIVE PASS: %zu ownership/atomicity checks; injected transport fixtures only\n", checks);
  return 0;
}
