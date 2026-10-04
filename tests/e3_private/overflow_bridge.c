/* The overflow-only tuple reuses the accepted retention observers. */
#include "retention_bridge.c"
#include <stdint.h>
#include <stdlib.h>

extern int64_t et_e3_test_published_counter_raw_v1(void *frame,
                                                   int64_t selector);

extern eshkol_tagged_value_t
et_e3_private_overflow_dispatch_cabi_v1(eshkol_tagged_value_t scenario,
                                        eshkol_tagged_value_t corpus_path);

void et_e3_overflow_run_v1(void *scenario, void *corpus_path, void *output) {
  et_e1b_ensure_private_initialized_v1();
  /* An alien pointer is compared against the registry before dereference. */
  if (et_e3_test_published_counter_raw_v1(NULL, 0) != -1 ||
      et_e3_test_published_counter_raw_v1((void *)(uintptr_t)1, 0) != -1)
    abort();
  *et_e1b_box_value_v1(output) = et_e3_private_overflow_dispatch_cabi_v1(
      *et_e1b_box_value_v1(scenario), *et_e1b_box_value_v1(corpus_path));
}
