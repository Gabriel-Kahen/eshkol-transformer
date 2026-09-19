#include <stdint.h>

#include "d2_wave2_package_bridge.c"

extern int64_t et_d2_dataset_test_live_count_v1(void);
extern int64_t et_d2_batch_test_live_count_v1(void);
extern int64_t et_d2_batch_test_borrow_count_v1(void);
extern int64_t et_d2_test_owned_allocation_count_v1(void);
extern int64_t et_d2_exact_read_test_fd_live_count_v1(void);
extern int64_t et_d2_exact_read_test_fd_peak_count_v1(void);

/* Test-only selector; this bridge is admitted only by the exact test tuple. */
int64_t et_e1b_public_d2_test_resource_count_v1(int64_t selector) {
  switch (selector) {
  case 0:
    return et_d2_dataset_test_live_count_v1();
  case 1:
    return et_d2_batch_test_live_count_v1();
  case 2:
    return et_d2_batch_test_borrow_count_v1();
  case 3:
    return et_d2_test_owned_allocation_count_v1();
  case 4:
    return et_d2_exact_read_test_fd_live_count_v1();
  case 5:
    return et_d2_exact_read_test_fd_peak_count_v1();
  default:
    return -1;
  }
}
