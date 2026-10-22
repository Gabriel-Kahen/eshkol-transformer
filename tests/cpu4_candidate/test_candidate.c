#define ET_G3R_CANDIDATE_RETIRE_PRIVATE 1
#define ET_F32_TENSOR_TESTING 1
#define ET_G3C4_NATIVE_OWNER_PRIVATE 1
#define ET_G3C4_NATIVE_OWNER_TESTING 1
#define ET_G3C4_NATIVE_PINS_PRIVATE 1
#define ET_G3C4_CONTEXT_PRIVATE 1
#define ET_G3C4_ACTIVE_CALL_PRIVATE 1
#define ET_G3C4_GENERATOR_PRIVATE 1
#define ET_CPU4_TRAINING_PRIVATE 1
#define ET_CPU4_CANDIDATE_PRIVATE 1
#define ET_CPU4_CANDIDATE_TESTING 1
#define ET_P1_PRIVATE_API 1
#include "../../src/eshkol_transformer/m3_call_f32_integration.c"
#include "../../src/eshkol_transformer/g3c4_model_owner.c"
#include <stdio.h>

static size_t checks;
#define CHECK(x) do { ++checks; if (!(x)) { \
  fprintf(stderr, "FAIL line %d: %s (domain=%lld category=%lld code=%lld)\n", \
    __LINE__, #x, (long long)et_g3c4_error_state.domain, \
    (long long)et_g3c4_error_state.category, (long long)et_g3c4_error_state.code); \
  exit(1); } } while (0)
#define OK(x) CHECK((x) == 0)
#define REJECT(x) CHECK((x) != 0)
static void *p1_context;

static et_cpu4_candidate_internal *begin(void) {
  et_cpu4_candidate_internal *candidate = et_cpu4_private_candidate_begin_v1(1729);
  CHECK(candidate != NULL);
  CHECK(et_cpu4_private_candidate_owner_v1(candidate) == candidate->owner);
  OK(et_p1_private_candidate_construction_begin_v1(p1_context));
  void *scope = et_p1_private_result_ptr_v1(p1_context);
  CHECK(scope != NULL);
  OK(et_cpu4_private_candidate_bind_graph_v1(candidate, p1_context, scope));
  return candidate;
}
static et_cpu4_candidate_internal *sealed(void) {
  et_cpu4_candidate_internal *candidate = begin();
  et_g3c4_model_owner_internal *owner = candidate->owner;
  OK(et_p1_private_construction_module_create_v1(p1_context, candidate->p1_scope));
  for (int64_t i = 0; i < 14; ++i) {
    OK(et_p1_private_construction_handle_create_v1(p1_context, candidate->p1_scope));
    void *handle = et_p1_private_result_ptr_v1(p1_context);
    CHECK(handle != NULL);
    OK(et_g3c4_private_model_owner_bind_v1(owner, i, handle));
  }
  for (int64_t i = 0; i < 14; ++i)
    OK(et_g3c4_private_model_owner_initialize_v1(owner, i));
  OK(et_p1_private_construction_prepare_v1(p1_context, candidate->p1_scope));
  OK(et_g3c4_private_model_owner_prepare_seal_v1(owner));
  /* A C4 commit cannot publish around a still-prepared P1 ledger. */
  REJECT(et_g3c4_private_model_owner_commit_seal_v1(owner));
  CHECK(candidate->state == CPU4_CANDIDATE_CONSTRUCTING);
  OK(et_p1_private_construction_commit_prepared_v1(p1_context, candidate->p1_scope));
  OK(et_g3c4_private_model_owner_commit_seal_v1(owner));
  CHECK(candidate->state == CPU4_CANDIDATE_UNPUBLISHED);
  return candidate;
}
static void retire(et_cpu4_candidate_internal *candidate) {
  et_f32_test_live_counts_v1 before = {.struct_size = sizeof(before)};
  et_f32_test_live_counts_v1 after = {.struct_size = sizeof(after)};
  et_f32_test_retired_counts_v1 dead_before = {.struct_size = sizeof(dead_before)};
  et_f32_test_retired_counts_v1 dead_after = {.struct_size = sizeof(dead_after)};
  et_f32_test_live_counts_snapshot_v1(&before);
  et_f32_test_retired_counts_snapshot_v1(&dead_before);
  OK(et_cpu4_private_candidate_retire_prepare_v1(candidate, NULL, NULL));
  OK(et_cpu4_private_candidate_retire_graph_commit_v1(candidate));
  CHECK(candidate->state == CPU4_CANDIDATE_REVOKED);
  for (size_t i = 0; i < 14u; ++i) CHECK(candidate->owner->parameters[i] != NULL);
  OK(et_cpu4_private_candidate_retire_payload_commit_v1(candidate));
  et_f32_test_live_counts_snapshot_v1(&after);
  et_f32_test_retired_counts_snapshot_v1(&dead_after);
  CHECK(before.parameters == after.parameters + 14u);
  CHECK(before.tensors == after.tensors + 28u);
  CHECK(dead_after.retained_control_bytes == dead_before.retained_control_bytes +
      14u * sizeof(et_f32_parameter) + 28u * sizeof(et_f32_tensor));
  CHECK(candidate->state == CPU4_CANDIDATE_DEAD);
  for (size_t i = 0; i < 14u; ++i) {
    CHECK(candidate->owner->parameters[i] == NULL);
    CHECK(candidate->owner->handles[i] == NULL);
    CHECK(et_p1_public_token_live_v1(candidate->handles[i]) == 0);
  }
  OK(et_cpu4_private_candidate_retire_payload_commit_v1(candidate));
}
static void *generator(et_g3c4_model_owner_internal *owner) {
  return et_g3c4_private_generator_seed_v1(owner, 9, 0,
      INT64_C(0x3f800000), 256, INT64_C(0x3f800000), 1, -1);
}
static et_o2_optimizer *optimizer(et_cpu4_candidate_internal *candidate) {
  et_o2_optimizer_builder *builder = NULL;
  et_o2_optimizer *result = NULL;
  et_o2_error_v1 error;
  OK(et_o2_optimizer_builder_create_v1(14u, ET_O2_CLIP_NONE, 0u,
      ET_O2_SCHEDULE_CONSTANT, 0u, 0u, UINT32_C(0x3f800000), &builder, &error));
  for (size_t i = 0; i < 14u; ++i)
    OK(et_o2_optimizer_builder_set_v1(builder, i,
        candidate->owner->parameters[i], candidate->owner->handles[i],
        UINT32_C(0x3ca3d70a), UINT32_C(0x3f4ccccd), UINT32_C(0x3f7d70a4),
        UINT32_C(0x322bcc77), 0u, &error));
  OK(et_o2_optimizer_builder_finish_v1(&builder, &result, &error));
  CHECK(result != NULL && builder == NULL);
  return result;
}
static void test_abort(void) {
  et_cpu4_candidate_internal *unbound = et_cpu4_private_candidate_begin_v1(1);
  CHECK(unbound != NULL);
  OK(et_cpu4_private_candidate_abort_open_v1(unbound));
  OK(et_cpu4_private_candidate_abort_open_v1(unbound));
  for (int prepared = 0; prepared < 2; ++prepared) {
    et_cpu4_candidate_internal *candidate = begin();
    REJECT(et_g3c4_private_model_owner_abort_v1(candidate->owner));
    REJECT(et_cpu4_private_candidate_abort_open_v1(candidate));
    CHECK(candidate->owner->parameters[0] != NULL);
    OK(et_p1_private_construction_module_create_v1(p1_context, candidate->p1_scope));
    if (prepared) {
      OK(et_p1_private_construction_prepare_v1(p1_context, candidate->p1_scope));
      OK(et_p1_private_construction_abort_prepared_v1(p1_context, candidate->p1_scope));
    } else OK(et_p1_private_construction_abort_v1(p1_context, candidate->p1_scope));
    OK(et_cpu4_private_candidate_abort_open_v1(candidate));
    CHECK(candidate->owner->parameters[0] == NULL);
    OK(et_cpu4_private_candidate_abort_open_v1(candidate));
  }
}
static void test_roster_and_retry(void) {
  et_cpu4_candidate_internal *a = sealed(), *b = sealed();
  et_cpu4_candidate_internal *cross = begin();
  REJECT(et_g3c4_private_model_owner_bind_v1(
      cross->owner, 0, (void *)b->owner->handles[0]));
  CHECK(cross->owner->handles[0] == NULL);
  OK(et_p1_private_construction_abort_v1(p1_context, cross->p1_scope));
  OK(et_cpu4_private_candidate_abort_open_v1(cross));
  et_g3c4_model_owner_internal *owner = a->owner;
  et_cpu4_candidate_internal copied = *a;
  REJECT(et_cpu4_private_candidate_retire_prepare_v1(&copied, NULL, NULL));
  REJECT(et_cpu4_private_candidate_retire_prepare_v1(owner, NULL, NULL));
  REJECT(et_cpu4_private_candidate_retire_payload_commit_v1(a));
  REJECT(et_cpu4_private_candidate_abort_open_v1(a));
  owner->candidate = (et_cpu4_candidate_internal *)1;
  CHECK(generator(owner) == NULL);
  REJECT(et_g3c4_private_model_owner_abort_v1(owner));
  owner->candidate = b;
  CHECK(et_g3c4_private_training_lease_create_v1(owner) == NULL);
  owner->candidate = NULL;
  CHECK(et_g3c4_private_context_create_v1(owner) == NULL);
  owner->candidate = a;
  a->state = CPU4_CANDIDATE_DEAD;
  REJECT(et_cpu4_private_candidate_abort_open_v1(a));
  REJECT(et_cpu4_private_candidate_retire_payload_commit_v1(a));
  CHECK(owner->parameters[0] != NULL);
  a->state = CPU4_CANDIDATE_UNPUBLISHED;
  et_f32_parameter *original = owner->parameters[13];
  owner->parameters[13] = b->owner->parameters[13];
  REJECT(et_cpu4_private_candidate_retire_prepare_v1(a, NULL, NULL));
  owner->parameters[13] = original;
  const void *handle = owner->handles[13];
  owner->handles[13] = b->owner->handles[13];
  REJECT(et_cpu4_private_candidate_retire_prepare_v1(a, NULL, NULL));
  owner->handles[13] = handle;
  et_f32_tensor *gradient = original->gradient;
  original->gradient = b->owner->parameters[13]->gradient;
  REJECT(et_cpu4_private_candidate_retire_prepare_v1(a, NULL, NULL));
  original->gradient = gradient;
  et_f32_tensor_error error;
  for (int grad = 0; grad < 2; ++grad) {
    et_f32_tensor_borrow *borrow = NULL;
    if (grad) OK(et_f32_parameter_gradient_borrow_begin_v1(original, &borrow, &error));
    else OK(et_f32_parameter_value_borrow_begin_v1(original, &borrow, &error));
    REJECT(et_cpu4_private_candidate_retire_prepare_v1(a, NULL, NULL));
    CHECK(a->state == CPU4_CANDIDATE_UNPUBLISHED);
    OK(et_f32_tensor_borrow_end_v1(&borrow, &error));
  }
  for (int pin = 0; pin < 3; ++pin) {
    if (pin == 0) ++original->plan_pins;
    if (pin == 1) ++original->value->plan_pins;
    if (pin == 2) ++original->gradient->plan_pins;
    REJECT(et_cpu4_private_candidate_retire_prepare_v1(a, NULL, NULL));
    if (pin == 0) --original->plan_pins;
    if (pin == 1) --original->value->plan_pins;
    if (pin == 2) --original->gradient->plan_pins;
  }
  void *lease = et_g3c4_private_training_lease_create_v1(owner);
  CHECK(lease != NULL);
  REJECT(et_cpu4_private_candidate_retire_prepare_v1(a, NULL, NULL));
  OK(et_g3c4_private_training_lease_prepare_release_v1(lease));
  OK(et_g3c4_private_training_lease_commit_release_v1(lease));
  void *context = et_g3c4_private_context_create_v1(owner);
  CHECK(context != NULL);
  REJECT(et_cpu4_private_candidate_retire_prepare_v1(a, NULL, NULL));
  REJECT(et_cpu4_private_candidate_retire_prepare_v1(a, NULL, context));
  OK(et_g3c4_private_context_close_v1(context));
  OK(et_cpu4_private_candidate_retire_prepare_v1(a, NULL, NULL));
  CHECK(generator(owner) == NULL && et_g3c4_private_training_lease_create_v1(owner) == NULL);
  CHECK(et_g3c4_private_context_create_v1(owner) == NULL);
  OK(et_cpu4_private_candidate_retire_cancel_v1(a));
  REJECT(et_cpu4_private_candidate_retire_graph_commit_v1(a));
  retire(a); /* A before B preserves the independent graph. */
  CHECK(b->owner->parameters[13] != NULL);
  void *g = generator(b->owner);
  CHECK(g != NULL);
  OK(et_g3c4_private_generator_close_v1(g));
  retire(b);
  a = sealed(); b = sealed();
  retire(b); retire(a); /* Reverse order. */
}
static void test_owned_children(void) {
  et_cpu4_candidate_internal *a = sealed();
  void *g = generator(a->owner);
  CHECK(g != NULL);
  REJECT(et_cpu4_private_candidate_retire_prepare_v1(a, NULL, NULL));
  void *other = generator(a->owner);
  CHECK(other != NULL);
  REJECT(et_cpu4_private_candidate_retire_prepare_v1(a, NULL, g));
  OK(et_g3c4_private_generator_close_v1(other));
  et_a2_kv_cache_read_borrow *borrow = NULL;
  et_kernel_error cache_error;
  OK(et_a2_kv_cache_read_borrow_begin_v1(
      ((et_g3c4_context_internal *)g)->cache, &borrow, &cache_error));
  REJECT(et_cpu4_private_candidate_retire_prepare_v1(a, NULL, g));
  OK(et_a2_kv_cache_read_borrow_end_v1(&borrow, &cache_error));
  et_o2_optimizer *o = optimizer(a);
  REJECT(et_cpu4_private_candidate_retire_prepare_v1(a, NULL, g));
  OK(et_cpu4_private_candidate_retire_prepare_v1(a, o, g));
  OK(et_cpu4_private_candidate_retire_graph_commit_v1(a));
  CHECK(a->owner->live_contexts == 0u);
  OK(et_g3c4_private_generator_close_v1(g));
  OK(et_cpu4_private_candidate_retire_payload_commit_v1(a));
  et_o2_error_v1 error;
  REJECT(et_o2_private_candidate_retire_preflight_v1(o, &error));
}
static void test_adopt(void) {
  et_cpu4_candidate_internal *a = sealed();
  void *g = generator(a->owner);
  CHECK(g != NULL);
  REJECT(et_cpu4_private_candidate_adopt_v1(a, NULL));
  OK(et_cpu4_private_candidate_adopt_v1(a, g));
  CHECK(a->state == CPU4_CANDIDATE_ADOPTED);
  REJECT(et_cpu4_private_candidate_retire_prepare_v1(a, NULL, g));
  REJECT(et_cpu4_private_candidate_abort_open_v1(a));
  REJECT(et_cpu4_private_candidate_retire_payload_commit_v1(a));
  OK(et_g3c4_private_generator_close_v1(g));
  CHECK(a->owner->parameters[0] != NULL);
}
int main(int argc, char **argv) {
  p1_context = et_p1_private_context_create_v1();
  CHECK(p1_context != NULL);
  if (argc == 2) {
    et_cpu4_candidate_internal *a = sealed();
    void *g = generator(a->owner);
    CHECK(g != NULL);
    et_o2_optimizer *o = optimizer(a);
    if (strcmp(argv[1], "second-consumer") == 0) {
      CHECK(optimizer(a) != NULL);
      REJECT(et_cpu4_private_candidate_retire_prepare_v1(a, o, g));
      CHECK(a->state == CPU4_CANDIDATE_UNPUBLISHED && a->owner->parameters[0] != NULL);
      puts("CPU4 second optimizer consumer rejected before mutation");
      return 0;
    }
    OK(et_cpu4_private_candidate_retire_prepare_v1(a, o, g));
    cpu4_candidate_fail_after = atoi(argv[1]);
    OK(et_cpu4_private_candidate_retire_graph_commit_v1(a));
    OK(et_cpu4_private_candidate_retire_payload_commit_v1(a));
    return 99; /* Every declared cut must stop before publication. */
  }
  CHECK(argc == 1);
  CHECK(et_cpu4_private_candidate_owner_v1((void *)1) == NULL);
  test_abort(); test_roster_and_retry(); test_owned_children(); test_adopt();
  printf("CPU4 native candidate PASS: %zu checks; real P1 graph, ordered O2/P1/C4 retirement, adoption and retry\n", checks);
  return 0;
}
