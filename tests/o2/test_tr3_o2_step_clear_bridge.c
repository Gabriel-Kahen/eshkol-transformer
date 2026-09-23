#define ET_TR3_O2_STEP_CLEAR_BRIDGE 1
#include "tr3_o2_step_clear_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

struct et_o2_trainer_step_clear_plan {
  uint32_t marker;
};

static struct et_o2_trainer_step_clear_plan test_plan = {UINT32_C(0x7472336f)};
static size_t prepare_calls;
static uint64_t seen_updates;
static uint64_t seen_contributions;
static uint32_t seen_weight_bits;

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__,     \
              #condition);                                                     \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

int32_t et_o2_trainer_step_clear_prepare_v1(
    et_o2_optimizer *optimizer, uint64_t expected_completed_updates,
    uint64_t expected_contribution_count,
    uint32_t expected_normalization_weight_bits,
    et_o2_trainer_step_clear_plan **plan, et_o2_error_v1 *error) {
  (void)optimizer;
  (void)error;
  prepare_calls++;
  seen_updates = expected_completed_updates;
  seen_contributions = expected_contribution_count;
  seen_weight_bits = expected_normalization_weight_bits;
  *plan = &test_plan;
  return 0;
}

int32_t et_o2_trainer_step_clear_commit_v1(
    et_o2_trainer_step_clear_plan *plan, et_o2_error_v1 *error) {
  (void)error;
  return plan == &test_plan ? 0 : 1;
}

int32_t et_o2_trainer_step_clear_abort_v1(
    et_o2_trainer_step_clear_plan *plan, et_o2_error_v1 *error) {
  (void)error;
  return plan == &test_plan ? 0 : 1;
}

void *et_tr3_private_o2_trainer_step_clear_prepare_v1(void *, int64_t,
                                                       int64_t, int64_t);
int64_t et_tr3_private_o2_trainer_step_clear_commit_v1(void *);
int64_t et_tr3_private_o2_trainer_step_clear_abort_v1(void *);
int64_t et_o2_private_last_error_category_v1(void);

int main(void) {
  void *plan;
  plan = et_tr3_private_o2_trainer_step_clear_prepare_v1(
      NULL, INT64_MAX, 2, UINT32_C(0x40000000));
  CHECK(plan == NULL);
  CHECK(prepare_calls == 0u);
  CHECK(et_o2_private_last_error_category_v1() ==
        ET_O2_STATUS_INVALID_ARGUMENT);

  plan = et_tr3_private_o2_trainer_step_clear_prepare_v1(
      NULL, 7, 2, UINT32_C(0x40000000));
  CHECK(plan == &test_plan);
  CHECK(prepare_calls == 1u);
  CHECK(seen_updates == 7u);
  CHECK(seen_contributions == 2u);
  CHECK(seen_weight_bits == UINT32_C(0x40000000));
  CHECK(et_tr3_private_o2_trainer_step_clear_commit_v1(plan) == 0);
  CHECK(et_tr3_private_o2_trainer_step_clear_abort_v1(plan) == 0);
  return 0;
}
