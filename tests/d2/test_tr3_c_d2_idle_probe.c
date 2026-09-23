#include "../../native/d2_native.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int64_t et_tr3_c_d2_dataset_idle_preflight_v1(const void *owner);

#define CHECK(condition)                                                     \
  do {                                                                       \
    if (!(condition)) {                                                       \
      fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__,      \
              #condition);                                                    \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

int main(void) {
  int owner = 0;
  int foreign = 0;
  int64_t generation;
  int64_t allocations;

  CHECK(et_tr3_c_d2_dataset_idle_preflight_v1(NULL) ==
        ET_D2_NATIVE_STATUS_INVALID_ARGUMENT);
  CHECK(et_d2_batch_last_status_v1() ==
        ET_D2_NATIVE_STATUS_INVALID_ARGUMENT);
  CHECK(et_tr3_c_d2_dataset_idle_preflight_v1(&foreign) ==
        ET_D2_NATIVE_STATUS_INVALID_ARGUMENT);
  CHECK(et_d2_batch_last_status_v1() ==
        ET_D2_NATIVE_STATUS_INVALID_ARGUMENT);

  CHECK(et_d2_dataset_open_v1(&owner, 0) == ET_D2_NATIVE_STATUS_OK);
  allocations = et_d2_test_owned_allocation_count_v1();
  CHECK(et_tr3_c_d2_dataset_idle_preflight_v1(&owner) ==
        ET_D2_NATIVE_STATUS_OK);
  CHECK(et_d2_batch_last_status_v1() == ET_D2_NATIVE_STATUS_OK);
  CHECK(et_d2_test_owned_allocation_count_v1() == allocations);

  generation = et_d2_batch_create_v1(&owner, 1, 1, 17);
  CHECK(generation > 0);
  allocations = et_d2_test_owned_allocation_count_v1();
  CHECK(et_tr3_c_d2_dataset_idle_preflight_v1(&owner) ==
        ET_D2_NATIVE_STATUS_INVALID_STATE);
  CHECK(et_d2_batch_last_status_v1() == ET_D2_NATIVE_STATUS_INVALID_STATE);
  CHECK(et_d2_test_owned_allocation_count_v1() == allocations);

  CHECK(et_d2_batch_release_preflight_v1(&owner, generation) ==
        ET_D2_NATIVE_STATUS_OK);
  CHECK(et_d2_batch_release_v1(&owner, generation) ==
        ET_D2_NATIVE_STATUS_OK);
  allocations = et_d2_test_owned_allocation_count_v1();
  CHECK(et_tr3_c_d2_dataset_idle_preflight_v1(&owner) ==
        ET_D2_NATIVE_STATUS_OK);
  CHECK(et_d2_batch_last_status_v1() == ET_D2_NATIVE_STATUS_OK);
  CHECK(et_d2_test_owned_allocation_count_v1() == allocations);

  CHECK(et_d2_dataset_close_v1(&owner) == ET_D2_NATIVE_STATUS_OK);
  CHECK(et_tr3_c_d2_dataset_idle_preflight_v1(&owner) ==
        ET_D2_NATIVE_STATUS_INVALID_ARGUMENT);
  CHECK(et_d2_batch_last_status_v1() ==
        ET_D2_NATIVE_STATUS_INVALID_ARGUMENT);
  CHECK(et_d2_test_owned_allocation_count_v1() == 0);

  puts("TR3-C D2 native idle probe PASS: registry, live batch, status, allocation");
  return 0;
}
