#include "t1_i64_shell.h"

#include "eshkol_transformer/i64_tensor.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int checks;

#define CHECK(condition)                                                     \
  do {                                                                       \
    ++checks;                                                                \
    if (!(condition)) {                                                      \
      fprintf(stderr, "T1 shell check failed at %s:%d: %s\n", __FILE__,     \
              __LINE__, #condition);                                         \
      return EXIT_FAILURE;                                                   \
    }                                                                        \
  } while (0)

static int check_rejected_candidate(void *candidate) {
  CHECK(et_t1_i64_shell_length_v1(candidate) == -1);
  CHECK(et_t1_i64_shell_last_status_v1() ==
        ET_T1_I64_SHELL_STATUS_INVALID_ARGUMENT);
  CHECK(et_t1_i64_shell_write_v1(candidate, 0, 1) ==
        ET_T1_I64_SHELL_STATUS_INVALID_ARGUMENT);
  CHECK(et_t1_i64_shell_read_v1(candidate, 0) == INT64_MIN);
  CHECK(et_t1_i64_shell_last_status_v1() ==
        ET_T1_I64_SHELL_STATUS_INVALID_ARGUMENT);
  CHECK(et_t1_i64_shell_seal_v1(candidate) ==
        ET_T1_I64_SHELL_STATUS_INVALID_ARGUMENT);
  CHECK(et_t1_i64_shell_abort_v1(candidate) ==
        ET_T1_I64_SHELL_STATUS_INVALID_ARGUMENT);
  return EXIT_SUCCESS;
}

int main(void) {
  static const int64_t exact_values[] = {
      INT64_MIN, -INT64_C(9007199254740993), 0,
      INT64_C(9007199254740993), INT64_MAX};
  int unrelated = 0;
  void *empty;
  void *exact;
  void *aborted;
  void *atomic;
  int64_t initial_live;
  size_t allowed;
  int saw_i1_success = 0;

  initial_live = et_t1_i64_shell_test_live_count_v1();
  CHECK(initial_live == 0);

  CHECK(et_t1_i64_shell_create_v1(-1) == NULL);
  CHECK(et_t1_i64_shell_last_status_v1() ==
        ET_T1_I64_SHELL_STATUS_INVALID_ARGUMENT);
  CHECK(et_t1_i64_shell_create_v1(INT64_MAX) == NULL);
  CHECK(et_t1_i64_shell_last_status_v1() == ET_T1_I64_SHELL_STATUS_RANGE);
  CHECK(et_t1_i64_shell_test_live_count_v1() == initial_live);

  et_t1_i64_shell_test_fail_stage_v1(
      ET_T1_I64_SHELL_TEST_FAIL_SHELL_ALLOC);
  CHECK(et_t1_i64_shell_create_v1(4) == NULL);
  CHECK(et_t1_i64_shell_last_status_v1() ==
        ET_T1_I64_SHELL_STATUS_ALLOCATION_FAILED);
  CHECK(et_t1_i64_shell_test_live_count_v1() == initial_live);

  et_t1_i64_shell_test_fail_stage_v1(
      ET_T1_I64_SHELL_TEST_FAIL_AFTER_TENSOR);
  CHECK(et_t1_i64_shell_create_v1(4) == NULL);
  CHECK(et_t1_i64_shell_last_status_v1() ==
        ET_T1_I64_SHELL_STATUS_INTERNAL);
  CHECK(et_t1_i64_shell_test_live_count_v1() == initial_live);

  et_t1_i64_shell_test_fail_stage_v1(
      ET_T1_I64_SHELL_TEST_FAIL_AFTER_BORROW);
  CHECK(et_t1_i64_shell_create_v1(4) == NULL);
  CHECK(et_t1_i64_shell_last_status_v1() ==
        ET_T1_I64_SHELL_STATUS_INTERNAL);
  CHECK(et_t1_i64_shell_test_live_count_v1() == initial_live);
  et_t1_i64_shell_test_fail_stage_v1(ET_T1_I64_SHELL_TEST_FAIL_NONE);

  /* Exercise every I1 allocation site and prove unpublished cleanup. */
  for (allowed = 0; allowed <= 8; ++allowed) {
    void *candidate;
    et_i64_tensor_test_fail_alloc_after_v1(allowed);
    candidate = et_t1_i64_shell_create_v1(4);
    if (candidate == NULL) {
      CHECK(et_t1_i64_shell_last_status_v1() ==
            ET_T1_I64_SHELL_STATUS_ALLOCATION_FAILED);
      CHECK(et_t1_i64_shell_test_live_count_v1() == initial_live);
    } else {
      saw_i1_success = 1;
      CHECK(et_t1_i64_shell_test_live_count_v1() == initial_live + 1);
      CHECK(et_t1_i64_shell_abort_v1(candidate) ==
            ET_T1_I64_SHELL_STATUS_OK);
      CHECK(et_t1_i64_shell_test_live_count_v1() == initial_live);
      break;
    }
  }
  et_i64_tensor_test_reset_allocator_v1();
  CHECK(saw_i1_success);

  /* None of these pointer values may be read before registry admission. */
  CHECK(check_rejected_candidate(NULL) == EXIT_SUCCESS);
  CHECK(check_rejected_candidate(&unrelated) == EXIT_SUCCESS);
  CHECK(check_rejected_candidate((void *)(uintptr_t)1) == EXIT_SUCCESS);
  CHECK(et_t1_i64_shell_test_live_count_v1() == initial_live);

  empty = et_t1_i64_shell_create_v1(0);
  CHECK(empty != NULL);
  CHECK(et_t1_i64_shell_test_live_count_v1() == initial_live + 1);
  CHECK(et_t1_i64_shell_length_v1(empty) == 0);
  CHECK(et_t1_i64_shell_last_status_v1() == ET_T1_I64_SHELL_STATUS_OK);
  CHECK(et_t1_i64_shell_write_v1(empty, 0, 7) ==
        ET_T1_I64_SHELL_STATUS_RANGE);
  CHECK(et_t1_i64_shell_read_v1(empty, 0) == INT64_MIN);
  CHECK(et_t1_i64_shell_last_status_v1() ==
        ET_T1_I64_SHELL_STATUS_INVALID_STATE);
  CHECK(et_t1_i64_shell_seal_v1(empty) == ET_T1_I64_SHELL_STATUS_OK);
  CHECK(et_t1_i64_shell_read_v1(empty, 0) == INT64_MIN);
  CHECK(et_t1_i64_shell_last_status_v1() == ET_T1_I64_SHELL_STATUS_RANGE);
  CHECK(et_t1_i64_shell_write_v1(empty, 0, 7) ==
        ET_T1_I64_SHELL_STATUS_INVALID_STATE);
  CHECK(et_t1_i64_shell_seal_v1(empty) ==
        ET_T1_I64_SHELL_STATUS_INVALID_STATE);
  CHECK(et_t1_i64_shell_abort_v1(empty) ==
        ET_T1_I64_SHELL_STATUS_INVALID_STATE);

  exact = et_t1_i64_shell_create_v1(
      (int64_t)(sizeof(exact_values) / sizeof(exact_values[0])));
  CHECK(exact != NULL);
  CHECK(et_t1_i64_shell_read_v1(exact, 0) == INT64_MIN);
  CHECK(et_t1_i64_shell_last_status_v1() ==
        ET_T1_I64_SHELL_STATUS_INVALID_STATE);
  for (allowed = 0; allowed < sizeof(exact_values) / sizeof(exact_values[0]);
       ++allowed) {
    CHECK(et_t1_i64_shell_write_v1(exact, (int64_t)allowed,
                                    exact_values[allowed]) ==
          ET_T1_I64_SHELL_STATUS_OK);
  }
  CHECK(et_t1_i64_shell_write_v1(exact, -1, 0) ==
        ET_T1_I64_SHELL_STATUS_RANGE);
  CHECK(et_t1_i64_shell_write_v1(
            exact, (int64_t)(sizeof(exact_values) / sizeof(exact_values[0])),
            0) == ET_T1_I64_SHELL_STATUS_RANGE);
  CHECK(et_t1_i64_shell_seal_v1(exact) == ET_T1_I64_SHELL_STATUS_OK);
  for (allowed = 0; allowed < sizeof(exact_values) / sizeof(exact_values[0]);
       ++allowed) {
    CHECK(et_t1_i64_shell_read_v1(exact, (int64_t)allowed) ==
          exact_values[allowed]);
    CHECK(et_t1_i64_shell_last_status_v1() == ET_T1_I64_SHELL_STATUS_OK);
  }
  CHECK(et_t1_i64_shell_read_v1(exact, -1) == INT64_MIN);
  CHECK(et_t1_i64_shell_last_status_v1() == ET_T1_I64_SHELL_STATUS_RANGE);
  CHECK(et_t1_i64_shell_write_v1(exact, 0, 1) ==
        ET_T1_I64_SHELL_STATUS_INVALID_STATE);

  aborted = et_t1_i64_shell_create_v1(1);
  CHECK(aborted != NULL);
  CHECK(et_t1_i64_shell_write_v1(aborted, 0, 23) ==
        ET_T1_I64_SHELL_STATUS_OK);
  CHECK(et_t1_i64_shell_abort_v1(aborted) == ET_T1_I64_SHELL_STATUS_OK);
  CHECK(et_t1_i64_shell_test_live_count_v1() == initial_live + 2);
  CHECK(check_rejected_candidate(aborted) == EXIT_SUCCESS);

  atomic = et_t1_i64_shell_create_v1(1);
  CHECK(atomic != NULL);
  CHECK(et_t1_i64_shell_write_v1(atomic, 0, INT64_MAX) ==
        ET_T1_I64_SHELL_STATUS_OK);
  CHECK(et_t1_i64_shell_write_v1(atomic, 1, INT64_MIN) ==
        ET_T1_I64_SHELL_STATUS_RANGE);
  CHECK(et_t1_i64_shell_seal_v1(atomic) == ET_T1_I64_SHELL_STATUS_OK);
  CHECK(et_t1_i64_shell_read_v1(atomic, 0) == INT64_MAX);
  CHECK(et_t1_i64_shell_last_status_v1() == ET_T1_I64_SHELL_STATUS_OK);
  CHECK(et_t1_i64_shell_test_live_count_v1() == initial_live + 3);

  printf("T1 I64 SHELL PASS: %d admission, exact-i64, lifetime, and failure checks\n",
         checks);
  return EXIT_SUCCESS;
}
