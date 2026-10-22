#define ET_F32_TENSOR_TESTING 1
#define ET_G3C4_NATIVE_OWNER_PRIVATE 1
#define ET_G3C4_NATIVE_OWNER_TESTING 1
#define ET_G3C4_NATIVE_PINS_PRIVATE 1
#define ET_G3C4_CONTEXT_PRIVATE 1
#define ET_G3C4_ACTIVE_CALL_PRIVATE 1
#define ET_G3C4_GENERATOR_PRIVATE 1
#define ET_CPU4_TRAINING_PRIVATE 1
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
static unsigned char identities[3][14], children[12], foreign;
typedef struct snapshot {
  et_f32_parameter *parameters[14];
  const void *handles[14];
  uint32_t values[14][1024], gradients[14][1024];
  et_f32_gradient_metadata_v1 metadata[14];
} snapshot;

static et_g3c4_model_owner_internal *model(size_t row) {
  et_g3c4_model_owner_internal *owner = et_g3c4_private_model_owner_create_seeded_v1(1729);
  CHECK(owner != NULL);
  for (int64_t i = 0; i < 14; ++i) {
    OK(et_g3c4_private_model_owner_bind_v1(owner, i, &identities[row][i]));
    OK(et_g3c4_private_model_owner_initialize_v1(owner, i));
  }
  OK(et_g3c4_private_model_owner_prepare_seal_v1(owner));
  OK(et_g3c4_private_model_owner_commit_seal_v1(owner));
  return owner;
}
static void capture(et_g3c4_model_owner_internal *owner, snapshot *out) {
  memset(out, 0, sizeof(*out));
  memcpy(out->parameters, owner->parameters, sizeof(out->parameters));
  memcpy(out->handles, owner->handles, sizeof(out->handles));
  for (size_t i = 0; i < 14; ++i) {
    et_f32_tensor_error error;
    size_t count = et_g3c4_parameter_elements(i);
    OK(et_f32_tensor_copy_bits_to_v1(et_g3c4_parameter_value(owner, i), out->values[i], count, &error));
    const et_f32_tensor *grad = owner->parameters[i]->gradient;
    OK(et_f32_tensor_copy_bits_to_v1(grad, out->gradients[i], count, &error));
    out->metadata[i].struct_size = sizeof(out->metadata[i]);
    OK(et_f32_parameter_gradient_metadata_v1(owner->parameters[i], &out->metadata[i], &error));
  }
}
static void unchanged(et_g3c4_model_owner_internal *owner, const snapshot *before) {
  snapshot after;
  capture(owner, &after);
  CHECK(memcmp(before, &after, sizeof(after)) == 0);
}
static void *generator(et_g3c4_model_owner_internal *owner) {
  return et_g3c4_private_generator_seed_v1(owner, 7, 0, INT64_C(0x3f800000),
      256, INT64_C(0x3f800000), 1, -1);
}
int main(void) {
  et_g3c4_model_owner_internal *a = model(0), *b = model(1);
  void *unsealed = et_g3c4_private_model_owner_create_seeded_v1(1729);
  CHECK(unsealed != NULL);
  CHECK(et_g3c4_private_training_lease_create_v1(unsealed) == NULL);
  OK(et_g3c4_private_model_owner_abort_v1(unsealed));
  uint32_t grad[1024];
  for (size_t i = 0; i < 1024; ++i) grad[i] = UINT32_C(0x3f000000) + (uint32_t)i;
  for (size_t i = 0; i < 14; ++i) {
    et_f32_parameter_test_set_gradient_bits_v1(a->parameters[i], grad, et_g3c4_parameter_elements(i));
    et_f32_parameter_test_set_metadata_v1(a->parameters[i], ET_F32_GRADIENT_PRESENT, 3, UINT32_C(0x40400000));
  }
  snapshot before;
  capture(a, &before);
  CHECK(et_g3c4_private_training_lease_create_v1(&foreign) == NULL);
  REJECT(et_g3c4_private_training_lease_check_v1(&foreign));

  /* Both raw and idle generator controls exclude whole-trainer admission. */
  void *context = et_g3c4_private_context_create_v1(a);
  CHECK(context != NULL && a->live_contexts == 1);
  CHECK(et_g3c4_private_training_lease_create_v1(a) == NULL);
  ((et_g3c4_context_internal *)context)->transport.busy = 1;
  REJECT(et_g3c4_private_context_close_v1(context));
  CHECK(a->live_contexts == 1);
  ((et_g3c4_context_internal *)context)->transport.busy = 0;
  OK(et_g3c4_private_context_close_v1(context));
  OK(et_g3c4_private_context_close_v1(context));
  CHECK(a->live_contexts == 0);
  context = generator(a);
  CHECK(context != NULL && a->live_contexts == 1);
  CHECK(et_g3c4_private_training_lease_create_v1(a) == NULL);
  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  CHECK(a->active == context);
  CHECK(et_g3c4_private_training_lease_create_v1(a) == NULL);
  REJECT(et_g3c4_private_generator_close_v1(context));
  CHECK(a->live_contexts == 1);
  OK(et_g3c4_private_call_abort_v1(context));
  OK(et_g3c4_private_generator_close_v1(context));
  OK(et_g3c4_private_generator_close_v1(context));
  CHECK(a->live_contexts == 0);
  a->live_contexts = SIZE_MAX;
  CHECK(generator(a) == NULL && et_g3c4_private_context_create_v1(a) == NULL);
  CHECK(a->live_contexts == SIZE_MAX);
  a->live_contexts = 0;

  /* Failed setup publishes neither lease nor canonical-storage changes. */
  et_g3c4_allocation_limit = et_g3c4_successful_allocations;
  CHECK(et_g3c4_private_training_lease_create_v1(a) == NULL && a->training_lease == NULL);
  et_g3c4_allocation_limit = SIZE_MAX;
  et_g3c4_training_lease_internal *lease = et_g3c4_private_training_lease_create_v1(a);
  CHECK(lease != NULL && a->training_lease == lease && a->active == NULL);
  et_g3c4_training_lease_internal copied = *lease;
  REJECT(et_g3c4_private_training_lease_check_v1(&copied));
  lease->magic ^= UINT64_C(1);
  REJECT(et_g3c4_private_training_lease_check_v1(lease));
  lease->magic ^= UINT64_C(1);
  CHECK(et_g3c4_private_training_lease_create_v1(a) == NULL);
  CHECK(generator(a) == NULL && et_g3c4_private_context_create_v1(a) == NULL);
  void *rng = et_g3c4_private_rng_seed_v1(4);
  CHECK(rng != NULL);
  CHECK(et_g3c4_private_generator_rng_v1(a, rng, 0, INT64_C(0x3f800000),
      256, INT64_C(0x3f800000), 1, -1) == NULL);
  OK(et_g3c4_private_rng_release_v1(rng));
  for (int64_t i = 0; i < 14; ++i) {
    CHECK(et_g3c4_private_training_lease_parameter_v1(lease, i) == a->parameters[i]);
    CHECK(et_g3c4_private_training_lease_identity_v1(lease, i) == a->handles[i]);
  }
  CHECK(et_g3c4_private_training_lease_parameter_v1(lease, -1) == NULL);
  CHECK(et_g3c4_private_training_lease_identity_v1(lease, 14) == NULL);
  a->handles[0] = b->handles[0];
  REJECT(et_g3c4_private_training_lease_check_v1(lease));
  a->handles[0] = before.handles[0];
  a->parameters[13] = b->parameters[13];
  REJECT(et_g3c4_private_training_lease_check_v1(lease));
  a->parameters[13] = before.parameters[13];
  OK(et_g3c4_private_training_lease_check_v1(lease));

  REJECT(et_g3c4_private_training_lease_enter_v1(lease, 0));
  lease->ordinal = UINT64_MAX;
  REJECT(et_g3c4_private_training_lease_enter_v1(lease, 1));
  CHECK(lease->ordinal == UINT64_MAX && lease->phase == CPU4_LEASE_IDLE);
  lease->ordinal = 0;
  OK(et_g3c4_private_training_lease_enter_v1(lease, 1));
  CHECK(lease->ordinal == 1);
  REJECT(et_g3c4_private_training_lease_enter_v1(lease, 2));
  REJECT(et_g3c4_training_child_acquire_internal(lease, NULL, 1));
  REJECT(et_g3c4_training_child_acquire_internal(lease, &children[0], 6));
  et_g3c4_allocation_limit = et_g3c4_successful_allocations;
  REJECT(et_g3c4_training_child_acquire_internal(lease, &children[0], 1));
  CHECK(lease->children == NULL);
  et_g3c4_allocation_limit = SIZE_MAX;
  for (int64_t kind = 1; kind <= 5; ++kind)
    OK(et_g3c4_training_child_acquire_internal(lease, &children[kind], kind));
  void *other_lease = et_g3c4_private_training_lease_create_v1(b);
  CHECK(other_lease != NULL);
  OK(et_g3c4_private_training_lease_enter_v1(other_lease, 1));
  REJECT(et_g3c4_training_child_acquire_internal(other_lease, &children[1], 1));
  REJECT(et_g3c4_training_child_release_internal(other_lease, &children[1], 1));
  OK(et_g3c4_private_training_lease_leave_v1(other_lease));
  OK(et_g3c4_private_training_lease_prepare_release_v1(other_lease));
  OK(et_g3c4_private_training_lease_commit_release_v1(other_lease));
  REJECT(et_g3c4_private_training_lease_leave_v1(lease));
  REJECT(et_g3c4_private_training_lease_prepare_release_v1(lease));
  REJECT(et_g3c4_training_child_release_internal(lease, &children[1], 2));
  for (int64_t kind = 1; kind <= 5; ++kind)
    OK(et_g3c4_training_child_release_internal(lease, &children[kind], kind));
  REJECT(et_g3c4_training_child_release_internal(lease, &children[1], 1));
  OK(et_g3c4_private_training_lease_leave_v1(lease));
  OK(et_g3c4_private_training_lease_enter_v1(lease, 1));
  CHECK(lease->ordinal == 2);
  REJECT(et_g3c4_training_child_acquire_internal(lease, &children[1], 1));
  REJECT(et_g3c4_training_child_release_internal(lease, &children[1], 1));
  OK(et_g3c4_private_training_lease_leave_v1(lease));
  OK(et_g3c4_private_training_lease_enter_v1(lease, 2));
  REJECT(et_g3c4_training_child_acquire_internal(lease, &children[6], 1));
  OK(et_g3c4_private_training_lease_leave_v1(lease));

  et_f32_tensor_borrow *borrow = NULL;
  et_f32_tensor_error error;
  OK(et_f32_parameter_value_borrow_begin_v1(a->parameters[13], &borrow, &error));
  REJECT(et_g3c4_private_training_lease_prepare_release_v1(lease));
  CHECK(lease->phase == CPU4_LEASE_IDLE && a->training_lease == lease);
  OK(et_f32_tensor_borrow_end_v1(&borrow, &error));
  OK(et_f32_parameter_gradient_borrow_begin_v1(a->parameters[13], &borrow, &error));
  REJECT(et_g3c4_private_training_lease_prepare_release_v1(lease));
  CHECK(lease->phase == CPU4_LEASE_IDLE && a->training_lease == lease);
  OK(et_f32_tensor_borrow_end_v1(&borrow, &error));
  OK(et_g3c4_private_training_lease_prepare_release_v1(lease));
  REJECT(et_g3c4_private_training_lease_enter_v1(lease, 1));
  CHECK(generator(a) == NULL);
  OK(et_g3c4_private_training_lease_cancel_release_v1(lease));
  REJECT(et_g3c4_private_training_lease_commit_release_v1(lease));
  OK(et_g3c4_private_training_lease_prepare_release_v1(lease));
  OK(et_g3c4_private_training_lease_commit_release_v1(lease));
  OK(et_g3c4_private_training_lease_commit_release_v1(lease));
  CHECK(a->training_lease == NULL && a->active == NULL && a->live_contexts == 0);
  REJECT(et_g3c4_private_training_lease_check_v1(lease));
  CHECK(et_g3c4_private_training_lease_parameter_v1(lease, 0) == NULL);
  unchanged(a, &before);
  context = generator(a);
  CHECK(context != NULL && a->live_contexts == 1);
  OK(et_g3c4_private_generator_close_v1(context));
  unchanged(a, &before);
  printf("CPU4 training lease PASS: %zu checks; canonical values/gradients/identities preserved\n", checks);
  return 0;
}
