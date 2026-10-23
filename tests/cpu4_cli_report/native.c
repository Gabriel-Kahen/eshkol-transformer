/* Actual native scalar unit; no test flags or production observer hooks. */
#define ET_G3R_CANDIDATE_RETIRE_PRIVATE 1
#include "../../src/eshkol_transformer/cpu4_cli_scalar_private.h"
#include "../../src/eshkol_transformer/m3t_f32_scoped.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
static int checks;
#define CHECK(x) do { assert(x); ++checks; } while (0)
static void private_error(int category) {
  CHECK(et_cpu4_private_cli_scalar_last_error_ref_v1(0) == 0);
  CHECK(et_cpu4_private_cli_scalar_last_error_ref_v1(1) == category);
  CHECK(et_cpu4_private_cli_scalar_last_error_ref_v1(2) == category);
}
int main(void) {
  et_f32_tensor *scalar = et_cpu4_private_cli_scalar_create_v1();
  et_f32_tensor_error error = {0};
  CHECK(scalar != NULL);
  size_t rank = 9, elements = 0, bytes = 0;
  CHECK(et_f32_tensor_rank_v1(scalar, &rank, &error) == 0 && rank == 0);
  CHECK(et_f32_tensor_element_count_v1(scalar, &elements, &error) == 0 && elements == 1);
  CHECK(et_f32_tensor_byte_length_v1(scalar, &bytes, &error) == 0 && bytes == 4);
  const uint32_t patterns[] = {0, 0x80000000u, 0x3f800000u, 0x7f800000u,
    0xff800000u, 0x7fc00001u, 0x7fa12345u, 0xffffffffu, 1, 0x007fffffu};
  for (size_t i = 0; i < sizeof(patterns)/sizeof(patterns[0]); ++i) {
    CHECK(et_f32_tensor_copy_bits_from_v1(scalar, patterns+i, 1, &error) == 0);
    CHECK(et_cpu4_private_cli_scalar_read_bits_v1(scalar) == (int64_t)patterns[i]);
  }
  CHECK(et_cpu4_private_cli_scalar_read_bits_v1(NULL) == -1); private_error(1);
  CHECK(et_cpu4_private_cli_scalar_read_bits_v1((et_f32_tensor *)((char *)scalar+1)) == -1); private_error(1);
  uint64_t shape = 1;
  et_f32_tensor *foreign = NULL;
  CHECK(et_f32_tensor_create_v1(1, &shape, &foreign, &error) == 0);
  CHECK(et_cpu4_private_cli_scalar_read_bits_v1(foreign) == -1); private_error(1);
  CHECK(et_cpu4_private_cli_scalar_release_v1(foreign) == -1); private_error(1);
  CHECK(et_f32_tensor_destroy_v1(&foreign, &error) == 0);
  CHECK(et_f32_tensor_create_v1(0, NULL, &foreign, &error) == 0);
  CHECK(et_cpu4_private_cli_scalar_read_bits_v1(foreign) == -1); private_error(1);
  CHECK(et_f32_tensor_destroy_v1(&foreign, &error) == 0);
  CHECK(et_cpu4_private_cli_scalar_claim_v1(scalar, 9) == -1); private_error(4);
  CHECK(et_cpu4_private_cli_scalar_claim_v1(scalar, 2) == -1); private_error(2);
  CHECK(et_cpu4_private_cli_scalar_claim_v1(scalar, 1) == 0);
  CHECK(et_cpu4_private_cli_scalar_claim_v1(scalar, 1) == -1); private_error(2);
  CHECK(et_cpu4_private_cli_scalar_release_v1(scalar) == -1); private_error(2);
  CHECK(et_cpu4_private_cli_scalar_read_bits_v1(scalar) == 0x007fffff);
  CHECK(et_cpu4_private_cli_scalar_claim_v1(scalar, 2) == 0);
  et_f32_tensor_borrow *borrow = NULL;
  CHECK(et_f32_tensor_borrow_begin_v1(scalar, &borrow, &error) == 0);
  CHECK(et_cpu4_private_cli_scalar_read_bits_v1(scalar) == -1);
  CHECK(et_cpu4_private_cli_scalar_last_error_ref_v1(0) == 1);
  CHECK(et_cpu4_private_cli_scalar_last_error_ref_v1(1) == ET_F32_TENSOR_ERROR_INVALID_STATE);
  CHECK(et_cpu4_private_cli_scalar_last_error_ref_v1(2) == ET_F32_TENSOR_CODE_ACTIVE_BORROW);
  CHECK(et_cpu4_private_cli_scalar_release_v1(scalar) == -1);
  CHECK(et_f32_tensor_borrow_end_v1(&borrow, &error) == 0);
  et_f32_scoped_guard_internal scoped = {0};
  CHECK(et_f32_tensor_scoped_begin_internal(scalar, &scoped, &error) == 0);
  CHECK(et_cpu4_private_cli_scalar_claim_v1(scalar, 1) == -1);
  CHECK(et_cpu4_private_cli_scalar_read_bits_v1(scalar) == -1);
  CHECK(et_f32_tensor_scoped_end_internal(&scoped) == 0);
  et_f32_tensor *source = et_cpu4_private_cli_scalar_create_v1();
  et_f32_tensor_copy_assignment_v1 assignment = {sizeof(assignment), scalar, source};
  et_f32_tensor_copy_plan *plan = NULL;
  CHECK(et_f32_tensor_copy_plan_prepare_v1(1, &assignment, &plan, &error) == 0);
  CHECK(et_cpu4_private_cli_scalar_read_bits_v1(scalar) == -1);
  CHECK(et_cpu4_private_cli_scalar_release_v1(scalar) == -1);
  CHECK(et_f32_tensor_copy_plan_release_v1(&plan, &error) == 0);
  CHECK(et_cpu4_private_cli_scalar_read_bits_v1(scalar) == 0x007fffff);
  CHECK(et_cpu4_private_cli_scalar_last_error_ref_v1(0) == 0);
  CHECK(et_cpu4_private_cli_scalar_last_error_ref_v1(1) == 0);
  CHECK(et_cpu4_private_cli_scalar_last_error_ref_v1(2) == 0);
  CHECK(et_cpu4_private_cli_scalar_last_error_ref_v1(-1) == -1);
  CHECK(et_cpu4_private_cli_scalar_last_error_ref_v1(3) == -1);
  CHECK(et_cpu4_private_cli_scalar_release_v1(source) == 0);
  CHECK(et_cpu4_private_cli_scalar_release_v1(scalar) == 0);
  CHECK(et_cpu4_private_cli_scalar_release_v1(scalar) == 0);
  CHECK(et_cpu4_private_cli_scalar_read_bits_v1(scalar) == -1); private_error(2);
  CHECK(et_cpu4_private_cli_scalar_claim_v1(scalar, 1) == -1); private_error(2);
  printf("CPU4-CLI-SCALAR-NATIVE-PASS %d\n", checks);
  return 0;
}
