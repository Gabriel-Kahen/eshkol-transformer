#include "t1_i64_shell.h"
#include "eshkol_transformer/i64_tensor.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int checks;
#define CHECK(c) do { ++checks; if (!(c)) { \
  fprintf(stderr, "exact-pair check %d failed at %s:%d: %s\n", \
          checks, __FILE__, __LINE__, #c); return EXIT_FAILURE; \
} } while (0)

static int check_corruption(void *source, void *other, int32_t field,
                            uintptr_t value, int32_t expected) {
  int64_t output[2] = {-11, -12};
  int64_t other_words[2] = {0, 0};
  CHECK(et_t1_i64_shell_test_corrupt_v1(source, field, value) == 0);
  CHECK(et_t1_i64_shell_private_c4_read_v1(source, output) == expected);
  CHECK(output[0] == -11 && output[1] == -12);
  CHECK(et_t1_i64_shell_private_c4_read_v1(other, other_words) ==
        ET_T1_C4_READ_OK);
  CHECK(other_words[0] == 3 && other_words[1] == 4);
  CHECK(et_t1_i64_shell_test_restore_v1(source) == 0);
  CHECK(et_t1_i64_shell_test_restore_v1(source) == -1);
  CHECK(et_t1_i64_shell_private_c4_read_v1(source, output) ==
        ET_T1_C4_READ_OK);
  CHECK(output[0] == 1 && output[1] == 2);
  return EXIT_SUCCESS;
}

int main(void) {
  void *a = et_t1_i64_shell_create_v1(2);
  void *b = et_t1_i64_shell_create_v1(2);
  void *unsealed = et_t1_i64_shell_create_v1(2);
  void *empty = et_t1_i64_shell_create_v1(0);
  void *long_input = et_t1_i64_shell_create_v1(3);
  void *negative = et_t1_i64_shell_create_v1(1);
  void *large = et_t1_i64_shell_create_v1(1);
  int64_t output[2] = {-11, -12};
  int64_t foreign[2] = {7, 8};
  int outsider = 0;
  CHECK(a && b && unsealed && empty && long_input && negative && large);
  CHECK(et_t1_i64_shell_write_v1(a, 0, 1) == 0);
  CHECK(et_t1_i64_shell_write_v1(a, 1, 2) == 0);
  CHECK(et_t1_i64_shell_write_v1(b, 0, 3) == 0);
  CHECK(et_t1_i64_shell_write_v1(b, 1, 4) == 0);
  CHECK(et_t1_i64_shell_write_v1(negative, 0, -1) == 0);
  CHECK(et_t1_i64_shell_write_v1(large, 0, 256) == 0);
  CHECK(et_t1_i64_shell_seal_v1(a) == 0);
  CHECK(et_t1_i64_shell_seal_v1(b) == 0);
  CHECK(et_t1_i64_shell_seal_v1(empty) == 0);
  CHECK(et_t1_i64_shell_seal_v1(long_input) == 0);
  CHECK(et_t1_i64_shell_seal_v1(negative) == 0);
  CHECK(et_t1_i64_shell_seal_v1(large) == 0);
  CHECK(et_t1_i64_shell_private_c4_read_v1(a, output) == ET_T1_C4_READ_OK);
  CHECK(output[0] == 1 && output[1] == 2);
  CHECK(et_t1_i64_shell_private_c4_read_v1(unsealed, output) ==
        ET_T1_C4_READ_INVALID_STATE);
  CHECK(et_t1_i64_shell_private_c4_read_v1(empty, output) ==
        ET_T1_C4_READ_SHAPE);
  CHECK(et_t1_i64_shell_private_c4_read_v1(long_input, output) ==
        ET_T1_C4_READ_SHAPE);
  CHECK(et_t1_i64_shell_private_c4_read_v1(negative, output) ==
        ET_T1_C4_READ_RANGE);
  CHECK(et_t1_i64_shell_private_c4_read_v1(large, output) ==
        ET_T1_C4_READ_RANGE);
  CHECK(et_t1_i64_shell_private_c4_read_v1(&outsider, output) ==
        ET_T1_C4_READ_INVALID_ARGUMENT);
  CHECK(et_t1_i64_shell_private_c4_read_v1((void *)(uintptr_t)1, output) ==
        ET_T1_C4_READ_INVALID_ARGUMENT);
  CHECK(et_t1_i64_shell_private_c4_read_v1(a, NULL) ==
        ET_T1_C4_READ_INVALID_ARGUMENT);
  CHECK(output[0] == 1 && output[1] == 2);

  CHECK(check_corruption(a,b,ET_T1_TEST_CORRUPT_RANK,2,ET_T1_C4_READ_SHAPE)==0);
  CHECK(check_corruption(a,b,ET_T1_TEST_CORRUPT_SHAPE_POINTER,1,
                         ET_T1_C4_READ_SHAPE)==0);
  CHECK(check_corruption(a,b,ET_T1_TEST_CORRUPT_SHAPE_EXTENT,3,
                         ET_T1_C4_READ_SHAPE)==0);
  CHECK(check_corruption(a,b,ET_T1_TEST_CORRUPT_STRUCT_SIZE,0,
                         ET_T1_C4_READ_SHAPE)==0);
  CHECK(check_corruption(a,b,ET_T1_TEST_CORRUPT_BYTE_LENGTH,1,
                         ET_T1_C4_READ_SHAPE)==0);
  CHECK(check_corruption(a,b,ET_T1_TEST_CORRUPT_DTYPE,1,
                         ET_T1_C4_READ_DTYPE)==0);
  CHECK(check_corruption(a,b,ET_T1_TEST_CORRUPT_DEVICE,1,
                         ET_T1_C4_READ_DEVICE)==0);
  CHECK(check_corruption(a,b,ET_T1_TEST_CORRUPT_LAYOUT,99,
                         ET_T1_C4_READ_LAYOUT)==0);
  CHECK(check_corruption(a,b,ET_T1_TEST_CORRUPT_OFFSET,8,
                         ET_T1_C4_READ_LAYOUT)==0);
  CHECK(check_corruption(a,b,ET_T1_TEST_CORRUPT_DATA,(uintptr_t)foreign,
                         ET_T1_C4_READ_STORAGE)==0);
  CHECK(check_corruption(a,b,ET_T1_TEST_CORRUPT_TENSOR,1,
                         ET_T1_C4_READ_INVALID_STATE)==0);
  CHECK(check_corruption(a,b,ET_T1_TEST_CORRUPT_BORROW,1,
                         ET_T1_C4_READ_INVALID_STATE)==0);
  CHECK(check_corruption(a,b,ET_T1_TEST_CORRUPT_VIEW,1,
                         ET_T1_C4_READ_INVALID_STATE)==0);
  CHECK(check_corruption(a,b,ET_T1_TEST_CORRUPT_LENGTH,3,
                         ET_T1_C4_READ_INVALID_STATE)==0);
  CHECK(check_corruption(a,b,ET_T1_TEST_CORRUPT_WHOLE_PAIR,(uintptr_t)b,
                         ET_T1_C4_READ_INVALID_STATE)==0);
  CHECK(et_t1_i64_shell_test_corrupt_v1(a, ET_T1_TEST_CORRUPT_RANK, 2)==0);
  CHECK(et_t1_i64_shell_test_corrupt_v1(a, ET_T1_TEST_CORRUPT_DTYPE, 1)==-1);
  CHECK(et_t1_i64_shell_test_restore_v1(a)==0);
  CHECK(et_t1_i64_shell_abort_v1(unsealed)==0);
  CHECK(et_t1_i64_shell_private_c4_read_v1(unsealed,output)==
        ET_T1_C4_READ_INVALID_ARGUMENT);
  CHECK(et_t1_i64_shell_write_v1(a,0,9)==
        ET_T1_I64_SHELL_STATUS_INVALID_STATE);
  CHECK(et_t1_i64_shell_abort_v1(a)==ET_T1_I64_SHELL_STATUS_INVALID_STATE);

  for (size_t allowed = 0; allowed < 8; ++allowed) {
    void *candidate;
    et_i64_tensor_test_fail_alloc_after_v1(allowed);
    candidate = et_t1_i64_shell_create_v1(2);
    et_i64_tensor_test_reset_allocator_v1();
    if (candidate != NULL) {
      CHECK(et_t1_i64_shell_abort_v1(candidate)==0);
      break;
    }
    CHECK(et_t1_i64_shell_last_status_v1()==
          ET_T1_I64_SHELL_STATUS_ALLOCATION_FAILED);
    CHECK(et_t1_i64_shell_private_c4_read_v1(a,output)==ET_T1_C4_READ_OK);
    CHECK(output[0]==1 && output[1]==2);
  }
  printf("G3-C4 T1/I1 EXACT PAIR PASS: %d checks\n", checks);
  return EXIT_SUCCESS;
}
