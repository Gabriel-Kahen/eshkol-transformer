#include "f32_parameter_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

void *et_i2_private_copy_builder_create_v1(int64_t count);
int64_t et_i2_private_copy_builder_set_v1(
    void *builder, int64_t index, void *destination, int64_t destination_role,
    void *destination_handle, void *source, int64_t source_role,
    void *source_handle);
int64_t et_i2_private_copy_builder_prepare_v1(void *builder);
void *et_i2_private_owned_clone_v1(void *carrier, int64_t role,
                                   void *p1_handle);
int64_t et_i2_private_owned_clone_live_count_v1(void);
int64_t et_tr3_c_i2_copy_builder_commit_checked_v1(void *builder);
int64_t et_tr3_c_i2_copy_builder_abort_checked_v1(void *builder);
void et_tr3_c_i2_owned_release_checked_v1(void *owned);
void et_tr3_c_i2_test_fail_copy_commit_v1(int64_t enabled);
void et_tr3_c_i2_test_fail_copy_release_v1(int64_t enabled);
void et_tr3_c_i2_test_fail_owned_release_v1(int64_t enabled);

enum { CARRIER_TENSOR = 2, CARRIER_OWNED_CLONE = 3 };

static int checks;
static int failures;

#define CHECK(condition)                                                       \
  do {                                                                         \
    checks++;                                                                   \
    if (!(condition)) {                                                         \
      (void)fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,           \
                    #condition);                                                \
      failures++;                                                               \
    }                                                                           \
  } while (0)

#define CHILD_REQUIRE(condition)                                                \
  do {                                                                         \
    if (!(condition)) {                                                        \
      _Exit(90);                                                               \
    }                                                                          \
  } while (0)

static et_f32_tensor *tensor_create(uint32_t bits) {
  const uint64_t shape[] = {1u};
  et_f32_tensor_error error;
  et_f32_tensor *tensor = NULL;
  if (et_f32_tensor_create_v1(1u, shape, &tensor, &error) != 0 ||
      et_f32_tensor_copy_bits_from_v1(tensor, &bits, 1u, &error) != 0) {
    return NULL;
  }
  return tensor;
}

static et_f32_tensor *owned_create(uint32_t bits) {
  et_f32_tensor_error error;
  et_f32_tensor *source = tensor_create(bits);
  et_f32_tensor *owned;
  if (source == NULL) {
    return NULL;
  }
  owned = (et_f32_tensor *)et_i2_private_owned_clone_v1(
      source, CARRIER_TENSOR, NULL);
  if (owned == NULL ||
      et_f32_tensor_destroy_v1(&source, &error) != 0) {
    return NULL;
  }
  return owned;
}

static void *prepared_builder(et_f32_tensor *destination,
                              et_f32_tensor *source) {
  void *builder = et_i2_private_copy_builder_create_v1(1);
  if (builder == NULL ||
      et_i2_private_copy_builder_set_v1(
          builder, 0, destination, CARRIER_TENSOR, NULL, source,
          CARRIER_OWNED_CLONE, NULL) != 0 ||
      et_i2_private_copy_builder_prepare_v1(builder) != 0) {
    return NULL;
  }
  return builder;
}

static void expect_exit_134(void (*action)(void)) {
  int status = 0;
  pid_t child = fork();
  CHECK(child >= 0);
  if (child == 0) {
    action();
    _Exit(91);
  }
  if (child < 0) {
    return;
  }
  CHECK(waitpid(child, &status, 0) == child);
  CHECK(WIFEXITED(status));
  CHECK(WEXITSTATUS(status) == 134);
}

static void test_commit_and_abort(void) {
  et_f32_tensor_error error;
  et_f32_tensor *destination = tensor_create(UINT32_C(0x3f800000));
  et_f32_tensor *source = owned_create(UINT32_C(0x80000000));
  void *builder = prepared_builder(destination, source);
  uint32_t actual = 0u;

  CHECK(destination != NULL && source != NULL && builder != NULL);
  CHECK(et_tr3_c_i2_copy_builder_commit_checked_v1(builder) == 0);
  CHECK(et_f32_tensor_copy_bits_to_v1(destination, &actual, 1u, &error) == 0);
  CHECK(actual == UINT32_C(0x80000000));
  CHECK(et_tr3_c_i2_copy_builder_commit_checked_v1(builder) == -1);
  CHECK(et_tr3_c_i2_copy_builder_abort_checked_v1(builder) == -1);
  et_tr3_c_i2_owned_release_checked_v1(source);
  CHECK(et_f32_tensor_destroy_v1(&destination, &error) == 0);

  destination = tensor_create(UINT32_C(0x40000000));
  source = owned_create(UINT32_C(0x40400000));
  builder = prepared_builder(destination, source);
  CHECK(destination != NULL && source != NULL && builder != NULL);
  CHECK(et_tr3_c_i2_copy_builder_abort_checked_v1(builder) == 0);
  CHECK(et_f32_tensor_copy_bits_to_v1(destination, &actual, 1u, &error) == 0);
  CHECK(actual == UINT32_C(0x40000000));
  et_tr3_c_i2_owned_release_checked_v1(source);
  CHECK(et_f32_tensor_destroy_v1(&destination, &error) == 0);

  builder = et_i2_private_copy_builder_create_v1(1);
  CHECK(builder != NULL);
  CHECK(et_tr3_c_i2_copy_builder_abort_checked_v1(builder) == 0);
  CHECK(et_tr3_c_i2_copy_builder_abort_checked_v1(NULL) == -1);
  CHECK(et_tr3_c_i2_copy_builder_commit_checked_v1(
            (void *)(uintptr_t)1u) == -1);
}

static void test_owned_clone_accounting(void) {
  et_f32_tensor *first;
  et_f32_tensor *second;

  CHECK(et_i2_private_owned_clone_live_count_v1() == 0);
  first = owned_create(UINT32_C(0x3f800000));
  second = owned_create(UINT32_C(0x40000000));
  CHECK(first != NULL && second != NULL);
  CHECK(et_i2_private_owned_clone_live_count_v1() == 2);
  et_tr3_c_i2_owned_release_checked_v1(first);
  CHECK(et_i2_private_owned_clone_live_count_v1() == 1);
  et_tr3_c_i2_owned_release_checked_v1(second);
  CHECK(et_i2_private_owned_clone_live_count_v1() == 0);
}

static void fail_unprepared_commit(void) {
  void *builder = et_i2_private_copy_builder_create_v1(1);
  CHILD_REQUIRE(builder != NULL);
  (void)et_tr3_c_i2_copy_builder_commit_checked_v1(builder);
}

static void fail_child_commit(void) {
  et_f32_tensor *destination = tensor_create(0u);
  et_f32_tensor *source = owned_create(UINT32_C(0x3f800000));
  void *builder = prepared_builder(destination, source);
  CHILD_REQUIRE(builder != NULL);
  et_tr3_c_i2_test_fail_copy_commit_v1(1);
  (void)et_tr3_c_i2_copy_builder_commit_checked_v1(builder);
}

static void fail_commit_release(void) {
  et_f32_tensor *destination = tensor_create(0u);
  et_f32_tensor *source = owned_create(UINT32_C(0x3f800000));
  void *builder = prepared_builder(destination, source);
  CHILD_REQUIRE(builder != NULL);
  et_tr3_c_i2_test_fail_copy_release_v1(1);
  (void)et_tr3_c_i2_copy_builder_commit_checked_v1(builder);
}

static void fail_abort_release(void) {
  et_f32_tensor *destination = tensor_create(0u);
  et_f32_tensor *source = owned_create(UINT32_C(0x3f800000));
  void *builder = prepared_builder(destination, source);
  CHILD_REQUIRE(builder != NULL);
  et_tr3_c_i2_test_fail_copy_release_v1(1);
  (void)et_tr3_c_i2_copy_builder_abort_checked_v1(builder);
}

static void fail_owned_release(void) {
  et_f32_tensor *owned = owned_create(UINT32_C(0x3f800000));
  CHILD_REQUIRE(owned != NULL);
  et_tr3_c_i2_test_fail_owned_release_v1(1);
  et_tr3_c_i2_owned_release_checked_v1(owned);
}

static void fail_nonowned_release(void) {
  et_f32_tensor *ordinary = tensor_create(UINT32_C(0x3f800000));
  CHILD_REQUIRE(ordinary != NULL);
  et_tr3_c_i2_owned_release_checked_v1(ordinary);
}

static void fail_stale_owned_release(void) {
  et_f32_tensor *owned = owned_create(UINT32_C(0x3f800000));
  CHILD_REQUIRE(owned != NULL);
  et_tr3_c_i2_owned_release_checked_v1(owned);
  et_tr3_c_i2_owned_release_checked_v1(owned);
}

int main(void) {
  test_commit_and_abort();
  test_owned_clone_accounting();
  expect_exit_134(fail_unprepared_commit);
  expect_exit_134(fail_child_commit);
  expect_exit_134(fail_commit_release);
  expect_exit_134(fail_abort_release);
  expect_exit_134(fail_owned_release);
  expect_exit_134(fail_nonowned_release);
  expect_exit_134(fail_stale_owned_release);
  if (failures != 0) {
    (void)fprintf(stderr, "TR3-C checked I2 bridge FAIL: %d/%d checks\n",
                  failures, checks);
    return 1;
  }
  (void)printf("TR3-C checked I2 bridge PASS: %d checks\n", checks);
  return 0;
}
