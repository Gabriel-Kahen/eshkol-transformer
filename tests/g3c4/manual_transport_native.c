#define ET_G3C4_MANUAL_ROLE_STEP_TEST_MAIN \
  et_g3c4_manual_role_step_predecessor_main
#include "test_manual_role_step.c"
#undef ET_G3C4_MANUAL_ROLE_STEP_TEST_MAIN

int64_t et_g3c4_manual_transport_test_field_v1(void *candidate,
                                                 int64_t field) {
  et_g3c4_logits_internal *logits;
  et_g3c4_context_internal *context;
  et_f32_tensor_error f32_error;
  uint32_t bits[256];
  int64_t length;
  if (field == 3) {
    context = et_g3c4_admit_context(candidate);
    if (context == NULL ||
        et_g3c4_output_committed_cache_length(context, &length) != 0)
      return -1;
    return length;
  }
  logits = et_g3c4_admit_logits(candidate, 1);
  if (logits == NULL) return -1;
  if (field == 0) return (int64_t)logits->transport.state;
  if (field == 1) return logits->parent_ctx == NULL ? 1 : 0;
  if (field == 2) {
    if (logits->tensor == NULL ||
        et_f32_tensor_copy_bits_to_v1(logits->tensor, bits, 256u,
                                      &f32_error) != 0)
      return -1;
    return (int64_t)bits[0];
  }
  return -1;
}

int64_t et_g3c4_manual_transport_test_fail_after_v1(int64_t allowed) {
  if (allowed < 0) return -1;
  et_g3c4_context_allocation_limit = (size_t)allowed;
  et_g3c4_context_successful_allocations = 0u;
  return 0;
}

int64_t et_g3c4_manual_transport_test_fail_reset_v1(void) {
  et_g3c4_context_allocation_limit = SIZE_MAX;
  et_g3c4_context_successful_allocations = 0u;
  return 0;
}
