#include "d2_native.c"
#include "e3_d2_native.h"

int64_t et_e3_d2_dataset_idle_preflight_v1(const void *owner) {
  et_d2_dataset_control *dataset = owner == NULL ? NULL : find_dataset(owner);
  return status(dataset == NULL ? ET_D2_NATIVE_STATUS_INVALID_ARGUMENT
                : dataset->current_batch != NULL ? ET_D2_NATIVE_STATUS_INVALID_STATE
                                                 : ET_D2_NATIVE_STATUS_OK);
}
