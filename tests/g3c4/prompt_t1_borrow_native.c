#define ET_I64_TENSOR_TESTING 1
#define ET_M3_TESTING 1
#define ET_F32_TENSOR_TESTING 1
#define ET_M3_CALL_TESTING 1
#define ET_G3C4_NATIVE_OWNER_PRIVATE 1
#define ET_G3C4_NATIVE_OWNER_TESTING 1
#define ET_G3C4_NATIVE_PINS_PRIVATE 1
#define ET_G3C4_CONTEXT_PRIVATE 1
#define ET_G3C4_CONTEXT_TESTING 1
#define ET_G3C4_ACTIVE_CALL_PRIVATE 1
#define ET_G3C4_ACTIVE_CALL_TESTING 1
#define ET_G3C4_GENERATOR_PRIVATE 1
#define ET_G3C4_PROMPT_T1_BORROW_PRIVATE 1

#include "../../src/eshkol_transformer/m3_call_f32_integration.c"
#include "../../src/eshkol_transformer/g3c4_model_owner.c"

extern int64_t et_m3_test_i64_counts_v1(int64_t field);

static et_i64_tensor_borrow *et_g3c4_prompt_test_borrow;
static et_g3c4_input_internal *et_g3c4_prompt_test_borrow_owner;

int64_t et_g3c4_prompt_test_transport_count_v1(int64_t kind) {
  int64_t count = 0;
  et_g3c4_transport_header_internal *item;
  for (item = et_g3c4_transport_registry; item != NULL;
       item = item->registry_next)
    if ((int64_t)item->kind == kind) count++;
  return count;
}

int64_t et_g3c4_prompt_test_length_v1(void *candidate) {
  et_g3c4_error_reset_internal();
  et_g3c4_input_internal *input = et_g3c4_admit_input(candidate, 0);
  return input == NULL ? -1 : input->length;
}

int64_t et_g3c4_prompt_test_token_v1(void *candidate, int64_t index) {
  int64_t words[2] = {0, 0};
  et_i64_tensor_error error;
  et_g3c4_error_reset_internal();
  et_g3c4_input_internal *input = et_g3c4_admit_input(candidate, 0);
  if (input == NULL) return INT64_MIN;
  if (index < 0 || index >= input->length) {
    (void)et_g3c4_fail(ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_SELECTOR);
    return INT64_MIN;
  }
  if (et_g3c4_capture_i64(
          et_i64_tensor_copy_to_v1(
              input->tensor, words, (size_t)input->length, &error),
          &error) != 0)
    return INT64_MIN;
  return words[index];
}

int64_t et_g3c4_prompt_test_i1_count_v1(int64_t field) {
  return et_m3_test_i64_counts_v1(field);
}

int64_t et_g3c4_prompt_test_owner_fail_after_v1(int64_t allowed) {
  if (allowed < 0) return -1;
  et_g3c4_context_allocation_limit = (size_t)allowed;
  et_g3c4_context_successful_allocations = 0u;
  return 0;
}

int64_t et_g3c4_prompt_test_owner_fail_reset_v1(void) {
  et_g3c4_context_allocation_limit = SIZE_MAX;
  et_g3c4_context_successful_allocations = 0u;
  return 0;
}

int64_t et_g3c4_prompt_test_i1_fail_after_v1(int64_t allowed) {
  if (allowed < 0) return -1;
  et_i64_tensor_test_fail_alloc_after_v1((size_t)allowed);
  return 0;
}

int64_t et_g3c4_prompt_test_i1_fail_reset_v1(void) {
  et_i64_tensor_test_reset_allocator_v1();
  return 0;
}

int64_t et_g3c4_prompt_test_active_borrow_v1(
    void *candidate, int64_t active) {
  et_i64_tensor_error error;
  et_g3c4_error_reset_internal();
  if (active == 1) {
    et_g3c4_input_internal *input = et_g3c4_admit_input(candidate, 0);
    if (input == NULL) return et_g3c4_error_state.category;
    if (et_g3c4_prompt_test_borrow != NULL)
      return et_g3c4_fail(
          ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
    if (et_g3c4_capture_i64(
            et_i64_tensor_borrow_begin_v1(
                input->tensor, &et_g3c4_prompt_test_borrow, &error),
            &error) != 0)
      return et_g3c4_error_state.category;
    et_g3c4_prompt_test_borrow_owner = input;
    return 0;
  }
  if (active == 0) {
    et_g3c4_input_internal *input = et_g3c4_admit_input(candidate, 0);
    if (input == NULL) return et_g3c4_error_state.category;
    if (et_g3c4_prompt_test_borrow == NULL ||
        et_g3c4_prompt_test_borrow_owner != input)
      return et_g3c4_fail(
          ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
    if (et_g3c4_capture_i64(
            et_i64_tensor_borrow_end_v1(
                &et_g3c4_prompt_test_borrow, &error),
            &error) != 0)
      return et_g3c4_error_state.category;
    et_g3c4_prompt_test_borrow_owner = NULL;
    return 0;
  }
  return et_g3c4_fail(
      ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_SELECTOR);
}
