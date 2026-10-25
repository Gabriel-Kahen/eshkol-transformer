/* Intrusive source test; the delivered helper exposes status only. */
#define ET_G3C4_NATIVE_PINS_PRIVATE 1
#define ET_G3C4_NATIVE_OWNER_PRIVATE 1
#define ET_G3R_C4_OWNER_TOPOLOGY_PRIVATE 1
#define ET_G3R_C4_OWNER_ASSOCIATION_PRIVATE 1
#define ET_G3R_C4_OWNER_PARAMETER_LEASE_PRIVATE 1
#include "../../src/eshkol_transformer/m3_call_f32_integration.c"
#include "../../src/eshkol_transformer/g3c4_model_owner.c"

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d: %s\n", \
  __FILE__, __LINE__, #x); exit(1); } } while (0)
#define OK(x) CHECK((x) == 0)

static unsigned char identity[14];
static et_g3c4_model_owner_internal *owner;
static unsigned char bits[4768];
static int checks;

static void snapshot(void) {
  size_t cursor = 0;
  for (size_t i = 0; i < 14; ++i) {
    size_t bytes = et_g3c4_parameter_elements(i) * sizeof(uint32_t);
    memcpy(bits + cursor, owner->parameters[i]->value->data, bytes);
    cursor += bytes;
  }
  CHECK(cursor == sizeof(bits));
}

static void unchanged(void) {
  size_t cursor = 0;
  for (size_t i = 0; i < 14; ++i) {
    et_f32_parameter *p = owner->parameters[i];
    size_t bytes = et_g3c4_parameter_elements(i) * sizeof(uint32_t);
    CHECK(!memcmp(bits + cursor, p->value->data, bytes));
    CHECK(!p->plan_pins && !p->value->active_borrow &&
          !p->gradient->active_borrow && !p->value->plan_pins &&
          !p->gradient->plan_pins);
    cursor += bytes;
  }
  CHECK(cursor == sizeof(bits));
}

static void status(int64_t wanted, int64_t domain, int64_t code) {
  et_g3c4_model_owner_internal before = *owner;
  et_f32_gradient_metadata_v1 metadata[14];
  et_f32_tensor_error error;
  for (size_t i = 0; i < 14; ++i) {
    metadata[i] = (et_f32_gradient_metadata_v1){.struct_size=sizeof(metadata[i])};
    OK(et_f32_parameter_gradient_metadata_v1(
        owner->parameters[i], &metadata[i], &error));
  }
  int64_t got = et_g3c4_private_sealed_owner_topology_status_v1(owner);
  if (got != wanted || et_g3c4_private_last_error_domain_v1() != domain ||
      et_g3c4_private_last_error_code_v1() != code)
    fprintf(stderr, "got status/domain/code %ld/%ld/%ld, wanted %ld/%ld/%ld\n",
        (long)got, (long)et_g3c4_private_last_error_domain_v1(),
        (long)et_g3c4_private_last_error_code_v1(),
        (long)wanted, (long)domain, (long)code);
  CHECK(got == wanted);
  CHECK(et_g3c4_private_last_error_domain_v1() == domain);
  CHECK(et_g3c4_private_last_error_category_v1() == wanted);
  CHECK(et_g3c4_private_last_error_code_v1() == code);
  CHECK(!memcmp(owner, &before, sizeof(before)));
  for (size_t i = 0; i < 14; ++i) {
    et_f32_gradient_metadata_v1 after = {.struct_size=sizeof(after)};
    OK(et_f32_parameter_gradient_metadata_v1(
        owner->parameters[i], &after, &error));
    CHECK(after.state == metadata[i].state &&
          after.contribution_count == metadata[i].contribution_count &&
          after.normalization_weight_bits == metadata[i].normalization_weight_bits);
  }
  unchanged();
  ++checks;
}

static void binding(const void *candidate, int64_t index,
    const void *handle, const void *parameter,
    int64_t wanted, int64_t domain, int64_t code) {
  et_g3c4_model_owner_internal before = *owner;
  int64_t got = et_g3c4_private_sealed_owner_binding_status_v1(
      candidate, index, handle, parameter);
  CHECK(got == wanted);
  CHECK(et_g3c4_private_last_error_domain_v1() == domain);
  CHECK(et_g3c4_private_last_error_category_v1() == wanted);
  CHECK(et_g3c4_private_last_error_code_v1() == code);
  CHECK(!memcmp(owner, &before, sizeof(before)));
  unchanged(); ++checks;
}

static void lease_status(const void *candidate, int64_t wanted,
    int64_t domain, int64_t code) {
  et_g3c4_model_owner_internal before = *owner;
  int64_t got = et_g3c4_private_sealed_owner_parameter_lease_status_v1(candidate);
  CHECK(got == wanted);
  CHECK(et_g3c4_private_last_error_domain_v1() == domain);
  CHECK(et_g3c4_private_last_error_category_v1() == wanted);
  CHECK(et_g3c4_private_last_error_code_v1() == code);
  CHECK(!memcmp(owner, &before, sizeof(before)));
  ++checks;
}

int main(void) {
  owner = et_g3c4_private_model_owner_create_seeded_v1(1729);
  CHECK(owner);
  for (size_t i = 0; i < 14; ++i)
    OK(et_g3c4_private_model_owner_bind_v1(owner, (int64_t)i, &identity[i]));
  for (size_t i = 0; i < 14; ++i)
    OK(et_g3c4_private_model_owner_initialize_v1(owner, (int64_t)i));
  snapshot();
  status(ET_G3C4_INVALID_STATE, ET_G3C4_DOMAIN_C4,
         ET_G3C4_CODE_LIFECYCLE);
  lease_status(owner, ET_G3C4_INVALID_STATE, ET_G3C4_DOMAIN_C4,
      ET_G3C4_CODE_LIFECYCLE);
  OK(et_g3c4_private_model_owner_prepare_seal_v1(owner));
  status(ET_G3C4_INVALID_STATE, ET_G3C4_DOMAIN_C4,
         ET_G3C4_CODE_LIFECYCLE);
  lease_status(owner, ET_G3C4_INVALID_STATE, ET_G3C4_DOMAIN_C4,
      ET_G3C4_CODE_LIFECYCLE);
  OK(et_g3c4_private_model_owner_commit_seal_v1(owner));
  status(0, 0, 0);
  lease_status(owner, 0, 0, 0);
  lease_status(NULL, ET_G3C4_INVALID_ARGUMENT, ET_G3C4_DOMAIN_C4,
      ET_G3C4_CODE_IDENTITY);
  int foreign_lease = 0;
  lease_status(&foreign_lease, ET_G3C4_INVALID_ARGUMENT, ET_G3C4_DOMAIN_C4,
      ET_G3C4_CODE_IDENTITY);
  owner->active = &foreign_lease;
  lease_status(owner, ET_G3C4_INVALID_STATE, ET_G3C4_DOMAIN_C4,
      ET_G3C4_CODE_LIFECYCLE);
  owner->active = NULL;
  for (size_t i = 0; i < 14; ++i) {
    et_f32_tensor_error error;
    et_f32_tensor_borrow *borrow = NULL;
    OK(et_f32_parameter_value_borrow_begin_v1(owner->parameters[i], &borrow, &error));
    lease_status(owner, ET_F32_TENSOR_ERROR_INVALID_STATE,
        ET_G3C4_DOMAIN_I2, ET_F32_TENSOR_CODE_INVALID_HANDLE);
    CHECK(borrow != NULL);
    CHECK(owner->parameters[i]->value->active_borrow == borrow);
    OK(et_f32_tensor_borrow_end_v1(&borrow, &error));
    CHECK(borrow == NULL);
    et_f32_parameter_test_set_metadata_v1(owner->parameters[i],
        ET_F32_GRADIENT_PRESENT, 1u, UINT32_C(0x3f800000));
    OK(et_f32_parameter_gradient_borrow_begin_v1(owner->parameters[i], &borrow, &error));
    lease_status(owner, ET_F32_TENSOR_ERROR_INVALID_STATE,
        ET_G3C4_DOMAIN_I2, ET_F32_TENSOR_CODE_INVALID_HANDLE);
    CHECK(borrow != NULL);
    CHECK(owner->parameters[i]->gradient->active_borrow == borrow);
    OK(et_f32_tensor_borrow_end_v1(&borrow, &error));
    et_f32_parameter_test_set_metadata_v1(owner->parameters[i],
        ET_F32_GRADIENT_ABSENT, 0u, 0u);
    lease_status(owner, 0, 0, 0);
  }
  et_f32_tensor_error lease_error;
  et_f32_gradient_reset_plan *reset = NULL;
  OK(et_f32_gradient_reset_plan_prepare_v1(
      14, owner->parameters, &reset, &lease_error));
  lease_status(owner, ET_F32_TENSOR_ERROR_INVALID_STATE,
      ET_G3C4_DOMAIN_I2, ET_F32_TENSOR_CODE_INVALID_HANDLE);
  OK(et_f32_gradient_reset_plan_release_v1(&reset, &lease_error));
  lease_status(owner, 0, 0, 0);
  unchanged();

  CHECK(et_g3c4_private_sealed_owner_topology_status_v1(NULL) ==
        ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_C4 &&
        et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_IDENTITY);
  unchanged(); ++checks;
  int foreign = 0;
  CHECK(et_g3c4_private_sealed_owner_topology_status_v1(&foreign) ==
        ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_C4 &&
        et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_IDENTITY);
  unchanged(); ++checks;

  owner->active = &foreign;
  status(ET_G3C4_INVALID_STATE, ET_G3C4_DOMAIN_C4,
         ET_G3C4_CODE_LIFECYCLE);
  owner->active = NULL;
  owner->initialized = 13;
  status(ET_G3C4_INVALID_STATE, ET_G3C4_DOMAIN_C4,
         ET_G3C4_CODE_READINESS);
  owner->initialized = 14;
  owner->handles[0] = &foreign;
  status(ET_F32_TENSOR_ERROR_INVALID_STATE, ET_G3C4_DOMAIN_I2,
         ET_F32_TENSOR_CODE_INVALID_HANDLE);
  owner->handles[0] = &identity[0];
  owner->handles[1] = owner->handles[0];
  status(ET_F32_TENSOR_ERROR_INVALID_STATE, ET_G3C4_DOMAIN_I2,
         ET_F32_TENSOR_CODE_INVALID_HANDLE);
  owner->handles[1] = &identity[1];
  owner->parameters[13]->value->shape[0] = 2;
  status(ET_F32_TENSOR_ERROR_SHAPE_MISMATCH, ET_G3C4_DOMAIN_I2,
         ET_F32_TENSOR_CODE_INVALID_SHAPE);
  owner->parameters[13]->value->shape[0] = 4;

  for (size_t cut = 0; cut < 14; ++cut) {
    m3_call_test_fail_after = cut;
    m3_call_test_release_count = 0;
    status(ET_F32_TENSOR_ERROR_INTERNAL, ET_G3C4_DOMAIN_I2,
           ET_F32_TENSOR_CODE_ALLOCATION_FAILED);
    CHECK(m3_call_test_release_count == cut);
    for (size_t i = 0; i < cut; ++i)
      CHECK(m3_call_test_release_order[i] == cut - i - 1);
  }
  m3_call_test_fail_after = SIZE_MAX;
  status(0, 0, 0);
  for (int64_t index = 0; index < 14; ++index)
    binding(owner, index, &identity[index], owner->parameters[index], 0, 0, 0);
  binding(owner, -1, &identity[0], owner->parameters[0],
          ET_G3C4_INVALID_ARGUMENT, ET_G3C4_DOMAIN_C4,
          ET_G3C4_CODE_SELECTOR);
  binding(owner, 14, &identity[0], owner->parameters[0],
          ET_G3C4_INVALID_ARGUMENT, ET_G3C4_DOMAIN_C4,
          ET_G3C4_CODE_SELECTOR);
  binding(owner, 0, &identity[1], owner->parameters[0],
          ET_G3C4_INVALID_STATE, ET_G3C4_DOMAIN_C4,
          ET_G3C4_CODE_STALE_BINDING);
  binding(owner, 0, &identity[0], owner->parameters[1],
          ET_G3C4_INVALID_STATE, ET_G3C4_DOMAIN_C4,
          ET_G3C4_CODE_STALE_BINDING);
  binding(owner, 0, &identity[0], &foreign,
          ET_G3C4_INVALID_STATE, ET_G3C4_DOMAIN_C4,
          ET_G3C4_CODE_STALE_BINDING);
  binding(&foreign, 0, &identity[0], owner->parameters[0],
          ET_G3C4_INVALID_ARGUMENT, ET_G3C4_DOMAIN_C4,
          ET_G3C4_CODE_IDENTITY);
  owner->active = &foreign;
  binding(owner, 0, &identity[0], owner->parameters[0],
          ET_G3C4_INVALID_STATE, ET_G3C4_DOMAIN_C4,
          ET_G3C4_CODE_LIFECYCLE);
  owner->active = NULL;
  unsigned char other_identity[14];
  et_g3c4_model_owner_internal *other =
      et_g3c4_private_model_owner_create_seeded_v1(1730);
  CHECK(other);
  for (size_t i = 0; i < 14; ++i)
    OK(et_g3c4_private_model_owner_bind_v1(other, (int64_t)i,
                                           &other_identity[i]));
  for (size_t i = 0; i < 14; ++i)
    OK(et_g3c4_private_model_owner_initialize_v1(other, (int64_t)i));
  OK(et_g3c4_private_model_owner_prepare_seal_v1(other));
  OK(et_g3c4_private_model_owner_commit_seal_v1(other));
  binding(other, 0, &identity[0], owner->parameters[0],
          ET_G3C4_INVALID_STATE, ET_G3C4_DOMAIN_C4,
          ET_G3C4_CODE_STALE_BINDING);
  et_g3c4_model_owner_internal *aborted =
      et_g3c4_private_model_owner_create_seeded_v1(1731);
  CHECK(aborted);
  OK(et_g3c4_private_model_owner_abort_v1(aborted));
  lease_status(aborted, ET_G3C4_INVALID_STATE, ET_G3C4_DOMAIN_C4,
      ET_G3C4_CODE_LIFECYCLE);
  binding(aborted, 0, &identity[0], owner->parameters[0],
          ET_G3C4_INVALID_STATE, ET_G3C4_DOMAIN_C4,
          ET_G3C4_CODE_LIFECYCLE);
  printf("G3-R C4 sealed owner topology: PASS checks=%d\n", checks);
  return 0;
}
