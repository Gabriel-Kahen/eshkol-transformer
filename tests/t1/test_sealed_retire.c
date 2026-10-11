#include "t1_i64_shell.h"

#include "eshkol_transformer/i64_tensor.h"

#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#if !defined(ET_T1_SEALED_RELEASE_TEST_ONLY) || \
    !defined(ET_T1_I64_SHELL_TESTING) || \
    !defined(ET_G3C4_T1_I1_EXACT_PAIR_PRIVATE) || \
    !defined(ET_I64_TENSOR_STORAGE_QUERY_PRIVATE)
#error "sealed retirement test requires all private test gates"
#endif

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "T1 sealed retirement failed at %s:%d\n", __FILE__,   \
              __LINE__);                                                       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static void *pair(int64_t first, int64_t second) {
  void *shell = et_t1_i64_shell_create_v1(2);
  CHECK(shell != NULL);
  CHECK(et_t1_i64_shell_write_v1(shell, 0, first) == 0);
  CHECK(et_t1_i64_shell_write_v1(shell, 1, second) == 0);
  CHECK(et_t1_i64_shell_seal_v1(shell) == 0);
  return shell;
}

static void check_fail_stop(void *shell) {
  int channel[2];
  int status;
  char marker;
  pid_t child;
  struct rlimit no_core = {0, 0};
  CHECK(pipe(channel) == 0);
  child = fork();
  CHECK(child >= 0);
  if (child == 0) {
    (void)setrlimit(RLIMIT_CORE, &no_core);
    close(channel[0]);
    et_t1_i64_shell_test_fail_after_borrow_end_v1(1);
    (void)et_t1_i64_shell_test_release_sealed_v1(shell);
    (void)write(channel[1], "S", 1); /* A return is forbidden. */
    _exit(9);
  }
  close(channel[1]);
  CHECK(waitpid(child, &status, 0) == child);
  CHECK(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
  CHECK(read(channel[0], &marker, 1) == 0);
  close(channel[0]);
  CHECK(et_t1_i64_shell_read_v1(shell, 0) == 0);
}

int main(void) {
  int unrelated = 0;
  int64_t baseline = et_t1_i64_shell_test_live_count_v1();
  void *unsealed = et_t1_i64_shell_create_v1(2);
  void *first;
  void *second;
  void *successor;
  void *wrong_length;
  const void *storage;
  CHECK(unsealed != NULL);
  CHECK(et_t1_i64_shell_test_release_sealed_v1(NULL) ==
        ET_T1_I64_SHELL_STATUS_INVALID_ARGUMENT);
  CHECK(et_t1_i64_shell_test_release_sealed_v1(&unrelated) ==
        ET_T1_I64_SHELL_STATUS_INVALID_ARGUMENT);
  CHECK(et_t1_i64_shell_test_release_sealed_v1(unsealed) ==
        ET_T1_I64_SHELL_STATUS_INVALID_STATE);
  CHECK(et_t1_i64_shell_test_live_count_v1() == baseline + 1);
  CHECK(et_t1_i64_shell_abort_v1(unsealed) == 0);
  wrong_length = et_t1_i64_shell_create_v1(1);
  CHECK(wrong_length != NULL);
  CHECK(et_t1_i64_shell_seal_v1(wrong_length) == 0);
  CHECK(et_t1_i64_shell_test_release_sealed_v1(wrong_length) ==
        ET_T1_I64_SHELL_STATUS_INVALID_STATE);
  CHECK(et_t1_i64_shell_length_v1(wrong_length) == 1);
  first = pair(0, 255);
  second = pair(0, 1);
  CHECK(et_t1_i64_shell_test_live_count_v1() == baseline + 3);
  CHECK(et_t1_i64_shell_abort_v1(first) ==
        ET_T1_I64_SHELL_STATUS_INVALID_STATE);
  CHECK(et_t1_i64_shell_test_corrupt_v1(
            first, ET_T1_TEST_CORRUPT_WHOLE_PAIR,
            (uintptr_t)second) == 0);
  CHECK(et_t1_i64_shell_test_release_sealed_v1(first) ==
        ET_T1_I64_SHELL_STATUS_INVALID_STATE);
  CHECK(et_t1_i64_shell_test_restore_v1(first) == 0);
  CHECK(et_t1_i64_shell_test_corrupt_v1(
            first, ET_T1_TEST_CORRUPT_RANK, 2) == 0);
  CHECK(et_t1_i64_shell_test_release_sealed_v1(first) ==
        ET_T1_I64_SHELL_STATUS_INVALID_STATE);
  CHECK(et_t1_i64_shell_test_restore_v1(first) == 0);
  CHECK(et_t1_i64_shell_test_live_count_v1() == baseline + 3);
  CHECK(et_t1_i64_shell_read_v1(first, 1) == 255);
  storage = et_t1_i64_shell_test_storage_v1(first);
  CHECK(storage != NULL);
  CHECK(et_i64_tensor_private_storage_overlap_v1(storage, 1) == 1);
  CHECK(et_t1_i64_shell_test_release_sealed_v1(first) == 0);
  CHECK(et_t1_i64_shell_test_live_count_v1() == baseline + 2);
  CHECK(et_i64_tensor_private_storage_overlap_v1(storage, 1) == 0);
  CHECK(et_t1_i64_shell_test_storage_v1(first) == NULL);
  CHECK(et_t1_i64_shell_test_release_sealed_v1(first) ==
        ET_T1_I64_SHELL_STATUS_INVALID_ARGUMENT);
  CHECK(et_t1_i64_shell_length_v1(first) == -1);
  CHECK(et_t1_i64_shell_last_status_v1() ==
        ET_T1_I64_SHELL_STATUS_INVALID_ARGUMENT);
  successor = pair(7, 8);
  CHECK(successor != first);
  CHECK(et_t1_i64_shell_read_v1(first, 0) == INT64_MIN);
  CHECK(et_t1_i64_shell_last_status_v1() ==
        ET_T1_I64_SHELL_STATUS_INVALID_ARGUMENT);
  check_fail_stop(second);
  CHECK(et_t1_i64_shell_test_release_sealed_v1(second) == 0);
  CHECK(et_t1_i64_shell_test_release_sealed_v1(successor) == 0);
  CHECK(et_t1_i64_shell_test_live_count_v1() == baseline + 1);
  puts("T1-SEALED-RETIRE-NATIVE-PASS");
  return 0;
}
