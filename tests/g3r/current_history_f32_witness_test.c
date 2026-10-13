/* Native prerequisite only: original fixed14 F32 bit witnesses under pins. */
#define ET_F32_TENSOR_TESTING 1
#define ET_G3R_SAME_OWNER_SAVE_PRIVATE 1
#include "../../src/eshkol_transformer/m3_call_f32_integration.c"
#include <stdio.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
  fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
  return 1; \
} } while (0)

static const uint64_t shapes[14][2] = {
  {4,4}, {4,4}, {4,4}, {4,4}, {4,8}, {8,4}, {4,0},
  {4,0}, {4,0}, {4,0}, {256,4}, {4,0}, {4,0}, {2,4}
};

int main(void) {
  et_f32_tensor_error error;
  et_f32_parameter *parameters[14] = {0};
  const void *identities[14] = {0};
  unsigned char anchors[14] = {0};
  et_g3t_model_pins_internal pins = {0};
  et_f32_parameter_save_witness *witness[14] = {0};
  for (size_t i = 0; i < 14; ++i) {
    CHECK(shapes[i][0] == m3_call_shapes[i][0] &&
          shapes[i][1] == m3_call_shapes[i][1]);
    et_f32_tensor *initial = NULL;
    const size_t rank = shapes[i][1] ? 2u : 1u;
    CHECK(et_f32_tensor_create_v1(rank, shapes[i], &initial, &error) == 0);
    CHECK(et_f32_parameter_create_v1(initial, &parameters[i], &error) == 0);
    identities[i] = &anchors[i];
    CHECK(et_f32_parameter_bind_identity_v1(
        parameters[i], identities[i], &error) == 0);
    CHECK(et_f32_tensor_destroy_v1(&initial, &error) == 0);
  }
  CHECK(et_g3t_model_pins_begin_internal(
      parameters, identities, &pins, &error) == 0);
  et_f32_parameter_save_witness *invalid = NULL;
  uint64_t wrong_shape[2] = {4, 5};
  CHECK(et_f32_parameter_save_witness_begin_v1(
      parameters[0], identities[1], 2u, shapes[0], &invalid,
      &error) != 0 && invalid == NULL);
  CHECK(et_f32_parameter_save_witness_begin_v1(
      parameters[0], identities[0], 2u, wrong_shape, &invalid,
      &error) != 0 && invalid == NULL);
  for (size_t i = 0; i < 14; ++i) {
    const size_t rank = shapes[i][1] ? 2u : 1u;
    CHECK(et_f32_parameter_save_witness_begin_v1(
        parameters[i], identities[i], rank, shapes[i], &witness[i],
        &error) == 0);
    CHECK(et_f32_parameter_save_witness_check_v1(witness[i], &error) == 0);
  }

  /* Changed value bits and absent-gradient bits must both be observed. */
  float *value_data = parameters[0]->value->data;
  uint32_t value_bits;
  memcpy(&value_bits, value_data, sizeof(value_bits));
  value_bits ^= UINT32_C(0x80000000);
  memcpy(value_data, &value_bits, sizeof(value_bits));
  CHECK(et_f32_parameter_save_witness_check_v1(witness[0], &error) != 0);
  value_bits ^= UINT32_C(0x80000000);
  memcpy(value_data, &value_bits, sizeof(value_bits));
  CHECK(et_f32_parameter_save_witness_check_v1(witness[0], &error) == 0);
  const size_t gradient_count = parameters[1]->gradient->element_count;
  uint32_t gradient_bits[16] = {0};
  CHECK(gradient_count == 16);
  gradient_bits[0] = UINT32_C(0x80000000);
  et_f32_parameter_test_set_gradient_bits_v1(
      parameters[1], gradient_bits, gradient_count);
  CHECK(et_f32_parameter_save_witness_check_v1(witness[1], &error) != 0);
  gradient_bits[0] = 0;
  et_f32_parameter_test_set_gradient_bits_v1(
      parameters[1], gradient_bits, gradient_count);
  CHECK(et_f32_parameter_save_witness_check_v1(witness[1], &error) == 0);

  /* A failed second-clone allocation leaves no witness or leaked baseline. */
  et_f32_test_live_counts_v1 before = {.struct_size = sizeof(before)};
  et_f32_test_live_counts_v1 after = {.struct_size = sizeof(after)};
  et_f32_test_live_counts_snapshot_v1(&before);
  et_f32_parameter_save_witness *extra = NULL;
  et_f32_tensor_test_fail_alloc_after_v1(5u);
  CHECK(et_f32_parameter_save_witness_begin_v1(
      parameters[0], identities[0], 2u, shapes[0], &extra, &error) != 0);
  CHECK(extra == NULL && error.code == ET_F32_TENSOR_CODE_ALLOCATION_FAILED);
  et_f32_tensor_test_fail_alloc_after_v1(SIZE_MAX);
  et_f32_test_live_counts_snapshot_v1(&after);
  CHECK(before.tensors == after.tensors && before.borrows == after.borrows);
  CHECK(et_f32_parameter_save_witness_check_v1(witness[0], &error) == 0);

  /* End is also the abort/drain path after a failed check. */
  value_bits ^= 1u;
  memcpy(value_data, &value_bits, sizeof(value_bits));
  CHECK(et_f32_parameter_save_witness_check_v1(witness[0], &error) != 0);
  CHECK(et_f32_parameter_save_witness_end_v1(&witness[0], &error) == 0);
  value_bits ^= 1u;
  memcpy(value_data, &value_bits, sizeof(value_bits));
  for (size_t i = 1; i < 14; ++i)
    CHECK(et_f32_parameter_save_witness_end_v1(&witness[i], &error) == 0);
  et_g3t_model_pins_end_internal(&pins);
  for (size_t i = 0; i < 14; ++i)
    CHECK(et_f32_parameter_destroy_v1(&parameters[i], &error) == 0);
  puts("current-history F32 fixed14 witness PASS");
  return 0;
}
