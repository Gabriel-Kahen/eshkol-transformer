#include "t1_i64_shell.h"
#include "t1_published_release_private.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int checks;
#define CHECK(test) do { \
  ++checks; \
  if (!(test)) { \
    fprintf(stderr, "T1 release check %d failed: %s\n", checks, #test); \
    return EXIT_FAILURE; \
  } \
} while (0)

int main(void) {
  int foreign = 0;
  void *first;
  void *second;
  void *third;
  int64_t pair[2] = {-1, -1};

  CHECK(et_t1_i64_shell_private_release_preflight_v1(&foreign) ==
        ET_T1_I64_SHELL_STATUS_INVALID_ARGUMENT);
  CHECK(et_t1_i64_shell_private_release_v1(&foreign) ==
        ET_T1_I64_SHELL_STATUS_INVALID_ARGUMENT);
  CHECK(et_t1_i64_shell_test_live_count_v1() == 0);

  first = et_t1_i64_shell_create_v1(2);
  CHECK(first != NULL);
  CHECK(et_t1_i64_shell_private_release_preflight_v1(first) ==
        ET_T1_I64_SHELL_STATUS_INVALID_STATE);
  CHECK(et_t1_i64_shell_private_release_v1(first) ==
        ET_T1_I64_SHELL_STATUS_INVALID_STATE);
  CHECK(et_t1_i64_shell_test_live_count_v1() == 1);
  CHECK(et_t1_i64_shell_write_v1(first, 0, 42) == 0);
  CHECK(et_t1_i64_shell_write_v1(first, 1, 43) == 0);
  CHECK(et_t1_i64_shell_seal_v1(first) == 0);
  CHECK(et_t1_i64_shell_test_corrupt_v1(
            first, ET_T1_TEST_CORRUPT_VIEW, (uintptr_t)NULL) == 0);
  CHECK(et_t1_i64_shell_private_release_preflight_v1(first) ==
        ET_T1_I64_SHELL_STATUS_INVALID_STATE);
  CHECK(et_t1_i64_shell_test_live_count_v1() == 1);
  CHECK(et_t1_i64_shell_test_restore_v1(first) == 0);
  CHECK(et_t1_i64_shell_private_release_preflight_v1(first) == 0);
  CHECK(et_t1_i64_shell_private_c4_read_v1(first, pair) ==
        ET_T1_C4_READ_OK);
  CHECK(pair[0] == 42 && pair[1] == 43);

  et_t1_i64_shell_test_fail_release_after_borrow_v1(1);
  CHECK(et_t1_i64_shell_private_release_v1(first) ==
        ET_T1_I64_SHELL_STATUS_INTERNAL);
  CHECK(et_t1_i64_shell_test_live_count_v1() == 1);
  CHECK(et_t1_i64_shell_length_v1(first) == -1);
  CHECK(et_t1_i64_shell_last_status_v1() ==
        ET_T1_I64_SHELL_STATUS_INVALID_STATE);
  CHECK(et_t1_i64_shell_read_v1(first, 0) == INT64_MIN);
  CHECK(et_t1_i64_shell_last_status_v1() ==
        ET_T1_I64_SHELL_STATUS_INVALID_STATE);
  CHECK(et_t1_i64_shell_private_c4_read_v1(first, pair) ==
        ET_T1_C4_READ_INVALID_STATE);
  CHECK(et_t1_i64_shell_private_release_preflight_v1(first) == 0);
  CHECK(et_t1_i64_shell_private_release_v1(first) == 0);
  CHECK(et_t1_i64_shell_test_live_count_v1() == 0);
  CHECK(et_t1_i64_shell_private_release_v1(first) == 0);
  CHECK(et_t1_i64_shell_test_live_count_v1() == 0);

  second = et_t1_i64_shell_create_v1(1);
  third = et_t1_i64_shell_create_v1(2);
  CHECK(second != NULL && third != NULL);
  CHECK(second != first && third != first);
  CHECK(et_t1_i64_shell_write_v1(second, 0, 7) == 0);
  CHECK(et_t1_i64_shell_seal_v1(second) == 0);
  CHECK(et_t1_i64_shell_write_v1(third, 0, 8) == 0);
  CHECK(et_t1_i64_shell_write_v1(third, 1, 9) == 0);
  CHECK(et_t1_i64_shell_seal_v1(third) == 0);
  CHECK(et_t1_i64_shell_test_live_count_v1() == 2);
  CHECK(et_t1_i64_shell_read_v1(first, 0) == INT64_MIN);
  CHECK(et_t1_i64_shell_last_status_v1() ==
        ET_T1_I64_SHELL_STATUS_INVALID_STATE);
  CHECK(et_t1_i64_shell_private_release_v1(first) == 0);
  CHECK(et_t1_i64_shell_read_v1(second, 0) == 7);
  CHECK(et_t1_i64_shell_private_release_v1(second) == 0);
  CHECK(et_t1_i64_shell_private_release_v1(third) == 0);
  CHECK(et_t1_i64_shell_test_live_count_v1() == 0);
  printf("T1 PUBLISHED RELEASE PASS: %d checks\n", checks);
  return EXIT_SUCCESS;
}
