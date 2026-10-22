/* Compile this integration TU once, with native/kernel_abi.c. */
#define ET_CPU4_CANDIDATE_PRIVATE 1
#define ET_G3R_CANDIDATE_RETIRE_PRIVATE 1
#define ET_F32_TENSOR_TESTING 1
#include "../../src/eshkol_transformer/m3t_f32_integration.c"

#include <assert.h>
#include <stdio.h>

int main(void) {
  uint64_t shape[] = {80u};
  et_f32_tensor *initial = NULL;
  et_f32_parameter *parameter = NULL;
  et_f32_parameter *peer = NULL;
  et_f32_tensor_error error;
  assert(et_f32_tensor_create_v1(1u, shape, &initial, &error) == 0);
  assert(et_f32_parameter_create_v1(initial, &parameter, &error) == 0);
  assert(et_f32_parameter_create_v1(initial, &peer, &error) == 0);
  et_f32_tensor *children[] = {parameter->value, parameter->gradient};
  const size_t allocations = et_f32_tensor_test_successful_allocations_v1();
  assert(et_cpu4_parameter_destroy_preflight_internal(parameter, &error) == 0);
  assert(et_cpu4_parameter_destroy_preflight_internal(parameter, NULL) == 0);
  assert(et_f32_tensor_test_successful_allocations_v1() == allocations);
  /* Parameter-owned children cannot use standalone tensor-destroy admission. */
  assert(et_f32_tensor_private_destroy_ready_v1(children[0], &error) != 0);
  assert(et_f32_tensor_private_destroy_ready_v1(children[1], &error) != 0);
  assert(et_cpu4_parameter_destroy_preflight_internal(NULL, &error) != 0);
  assert(et_cpu4_parameter_destroy_preflight_internal((void *)1, &error) != 0);
  parameter->plan_pins = 1u;
  assert(et_cpu4_parameter_destroy_preflight_internal(parameter, &error) != 0);
  parameter->plan_pins = 0u;
  et_f32_tensor *peer_value = peer->value;
  peer->value = children[0];
  assert(et_cpu4_parameter_destroy_preflight_internal(parameter, &error) != 0);
  peer->value = peer_value;
  assert(et_cpu4_parameter_destroy_preflight_internal(parameter, &error) == 0);
  for (size_t child = 0u; child < 2u; ++child) {
    et_f32_tensor **slot = child == 0u ? &parameter->value : &parameter->gradient;
    et_f32_tensor_borrow *borrow = NULL;
    assert(et_f32_tensor_borrow_begin_v1(children[child], &borrow, &error) == 0);
    assert(et_cpu4_parameter_destroy_preflight_internal(parameter, &error) != 0);
    assert(parameter->value == children[0] && parameter->gradient == children[1]);
    assert(et_f32_tensor_borrow_end_v1(&borrow, &error) == 0);
    children[child]->plan_pins = 1u;
    assert(et_cpu4_parameter_destroy_preflight_internal(parameter, &error) != 0);
    children[child]->plan_pins = 0u;
    *slot = NULL;
    assert(et_cpu4_parameter_destroy_preflight_internal(parameter, &error) != 0);
    *slot = (void *)1;
    assert(et_cpu4_parameter_destroy_preflight_internal(parameter, &error) != 0);
    *slot = initial;
    assert(et_cpu4_parameter_destroy_preflight_internal(parameter, &error) != 0);
    *slot = children[1u - child];
    assert(et_cpu4_parameter_destroy_preflight_internal(parameter, &error) != 0);
    *slot = peer->value;
    assert(et_cpu4_parameter_destroy_preflight_internal(parameter, &error) != 0);
    *slot = peer->gradient;
    assert(et_cpu4_parameter_destroy_preflight_internal(parameter, &error) != 0);
    *slot = children[child];
    assert(et_cpu4_parameter_destroy_preflight_internal(peer, &error) == 0);
    et_f32_test_tensor_metadata_v1 metadata = {.struct_size = sizeof(metadata)};
    assert(et_f32_tensor_test_metadata_snapshot_v1(children[child], &metadata) == 0);
    et_f32_test_tensor_metadata_v1 forged = metadata;
    float decoy[80] = {0};
    unsigned char before[sizeof(error)];
    memcpy(before, metadata.data_storage, sizeof(before));
    forged.data_storage = decoy;
    assert(et_f32_tensor_test_metadata_restore_v1(children[child], &forged) == 0);
    assert(et_cpu4_parameter_destroy_preflight_internal(parameter, &error) != 0);
    assert(et_cpu4_parameter_destroy_preflight_internal(
        NULL, (et_f32_tensor_error *)metadata.data_storage) != 0);
    assert(memcmp(before, metadata.data_storage, sizeof(before)) == 0);
    assert(et_f32_tensor_test_metadata_restore_v1(children[child], &metadata) == 0);
    assert(et_cpu4_parameter_destroy_preflight_internal(parameter, &error) == 0);
  }
  et_f32_parameter *stale = parameter;
  assert(et_f32_parameter_destroy_v1(&parameter, &error) == 0);
  assert(parameter == NULL);
  assert(et_cpu4_parameter_destroy_preflight_internal(stale, &error) != 0);
  assert(et_cpu4_parameter_destroy_preflight_internal(peer, &error) == 0);
  assert(et_f32_parameter_destroy_v1(&peer, &error) == 0);
  assert(et_f32_tensor_destroy_v1(&initial, &error) == 0);
  puts("CPU4 parameter destroy preflight PASS: parameter-owned children, leases, provenance and retry");
  return 0;
}
