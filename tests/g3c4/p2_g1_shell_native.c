#define ET_G3C4_P2_G1_PENDING_PRIVATE 1
#define ET_G3C4_P2_G1_PUBLICATION_PRIVATE 1
#include "output_envelope_native.c"

typedef struct shell_bytevector {
  int64_t length;
  unsigned char bytes[8];
} shell_bytevector;

typedef struct shell_copyout_stage {
  int64_t length;
  unsigned char bytes[56];
} shell_copyout_stage;

static et_i64_tensor_borrow *shell_ids_borrow;
static et_a2_kv_cache_transaction_view *shell_a2_view;
static shell_bytevector *shell_staging;
static int64_t shell_mutation;
static int64_t copyout_corrupt;
static int64_t copyout_corrupt_consumed;
#ifdef ET_G3C4_P2_G1_COORDINATOR_PRIVATE
static et_i64_tensor_borrow *coordinator_input_borrow;
static et_a2_kv_cache_read_borrow *coordinator_cache_borrow;

int64_t et_g3c4_coordinator_test_prefill_token_v1(
    void *context_candidate, int64_t index) {
  et_g3c4_context_internal *context =
      et_g3c4_admit_idle_call(context_candidate);
  if (context == NULL || index < 0 || index > 1 ||
      context->prefill_binding_ready != 1u) return -1;
  return context->prefill_tokens[index];
}

int64_t et_g3c4_coordinator_test_input_borrow_v1(
    void *input_candidate, int64_t active) {
  et_g3c4_input_internal *input = et_g3c4_admit_input(input_candidate, 0);
  et_i64_tensor_error error;
  if (input == NULL) return -1;
  if (active == 1 && coordinator_input_borrow == NULL)
    return et_i64_tensor_borrow_begin_v1(
        input->tensor, &coordinator_input_borrow, &error);
  if (active == 0 && coordinator_input_borrow != NULL)
    return et_i64_tensor_borrow_end_v1(&coordinator_input_borrow, &error);
  return -1;
}

int64_t et_g3c4_coordinator_test_cache_borrow_v1(
    void *context_candidate, int64_t active) {
  et_g3c4_context_internal *context =
      et_g3c4_admit_idle_call(context_candidate);
  et_kernel_error error;
  if (context == NULL) return -1;
  if (active == 1 && coordinator_cache_borrow == NULL)
    return et_a2_kv_cache_read_borrow_begin_v1(
        context->cache, &coordinator_cache_borrow, &error);
  if (active == 0 && coordinator_cache_borrow != NULL)
    return et_a2_kv_cache_read_borrow_end_v1(
        &coordinator_cache_borrow, &error);
  return -1;
}
#endif

int64_t et_g3c4_shell_test_copyout_corrupt_v1(int64_t active) {
  if (active != 0 && active != 1) return -1;
  copyout_corrupt = active;
  copyout_corrupt_consumed = 0;
  return 0;
}

int64_t et_g3c4_shell_test_copyout_corrupt_consumed_v1(void) {
  return copyout_corrupt_consumed;
}

int64_t __real_et_g3c4_private_output_copy_snapshot_v1(
    void *, void *, void *);
int64_t __wrap_et_g3c4_private_output_copy_snapshot_v1(
    void *output, void *raw, void *stage) {
  int64_t status = __real_et_g3c4_private_output_copy_snapshot_v1(
      output, raw, stage);
  if (status == 0 && copyout_corrupt != 0) {
    shell_copyout_stage *carrier = (shell_copyout_stage *)stage;
    carrier->bytes[8] = 2u; /* Decoded length is invalid after native success. */
    for (size_t i = 9; i < 16; ++i) carrier->bytes[i] = 0u;
    copyout_corrupt = 0;
    copyout_corrupt_consumed = 1;
  }
  return status;
}

int64_t et_g3c4_shell_test_prepare_v1(void *context_candidate,
                                       void *output_candidate) {
  const int64_t tokens[2] = {0, 255};
  float prefill_logits[256], token_logits[256];
  int64_t token = -1;
  et_g3c4_context_internal *context =
      et_g3c4_admit_active_call(context_candidate);
  et_g3c4_output_internal *output =
      et_g3c4_admit_output(output_candidate, 0);
  if (context == NULL || output == NULL || output->parent_ctx != context ||
      output->prompt_length != 2 || output->generated_length != 1)
    return -1;
  et_g3c4_input_internal *input = create_prompt(tokens, 2);
  OK(et_g3c4_private_prompt_prefill_v1(context, input, prefill_logits));
  OK(et_g3c4_private_token_frame_begin_last_v1(
      context, prefill_logits, &token));
  OK(et_g3c4_private_token_forward_v1(
      context, token, token_logits));
  OK(et_g3c4_private_tensor_release_v1(input));
  return 0;
}

int64_t et_g3c4_shell_test_ids_borrow_v1(void *output_candidate,
                                          int64_t active) {
  et_g3c4_output_internal *output =
      (et_g3c4_output_internal *)output_candidate;
  et_i64_tensor_error error;
  if (output == NULL || output->transport.magic != ET_G3C4_OUTPUT_MAGIC ||
      output->ids == NULL) return -1;
  if (active == 1 && shell_ids_borrow == NULL)
    return et_i64_tensor_borrow_begin_v1(
        output->ids, &shell_ids_borrow, &error);
  if (active == 0 && shell_ids_borrow != NULL)
    return et_i64_tensor_borrow_end_v1(&shell_ids_borrow, &error);
  return -1;
}

int64_t et_g3c4_shell_test_a2_view_v1(void *context_candidate,
                                       int64_t active) {
  et_g3c4_context_internal *context =
      et_g3c4_admit_active_call(context_candidate);
  et_kernel_error error;
  if (context == NULL) return -1;
  if (active == 1 && shell_a2_view == NULL)
    return et_a2_kv_cache_transaction_view_begin_v1(
        context->token_frame_transaction, 0u, &shell_a2_view, &error);
  if (active == 0 && shell_a2_view != NULL)
    return et_a2_kv_cache_transaction_view_end_v1(&shell_a2_view, &error);
  return -1;
}

int64_t et_g3c4_shell_test_mutation_v1(int64_t mode) {
  if (mode < 0 || mode > 3) return -1;
  shell_mutation = mode;
  shell_staging = NULL;
  return 0;
}

int64_t et_g3c4_shell_test_published_v1(void *output_candidate,
                                         void *raw_candidate) {
  et_g3c4_output_internal *output =
      (et_g3c4_output_internal *)output_candidate;
  shell_bytevector *raw = (shell_bytevector *)raw_candidate;
  et_i64_tensor_borrow *borrow = NULL;
  et_i64_tensor_error error;
  const et_kernel_tensor_view_v1 *view = NULL;
  int64_t exact = 0;
  if (output == NULL || raw == NULL ||
      output->transport.magic != ET_G3C4_OUTPUT_MAGIC ||
      output->transport.state != ET_G3C4_OUTPUT_PUBLISHED ||
      output->length != 1 || output->cache_length != 3 ||
      raw->length != 1 || output->ids == NULL)
    return 0;
  if (et_i64_tensor_borrow_begin_v1(output->ids, &borrow, &error) != 0)
    return 0;
  if (et_i64_tensor_borrow_view_v1(borrow, &view, &error) == 0 &&
      view != NULL && view->data != NULL &&
      *(const int64_t *)view->data == (int64_t)raw->bytes[0]) exact = 1;
  if (et_i64_tensor_borrow_end_v1(&borrow, &error) != 0) return 0;
  return exact;
}

int64_t __real_et_g3c4_private_output_copy_decode_ids_v1(
    void *, void *, void *);
int64_t __wrap_et_g3c4_private_output_copy_decode_ids_v1(
    void *context, void *output, void *staging) {
  int64_t status = __real_et_g3c4_private_output_copy_decode_ids_v1(
      context, output, staging);
  if (status == 0) shell_staging = (shell_bytevector *)staging;
  return status;
}

int64_t __real_et_g3c4_private_output_accept_text_v1(
    void *, void *, void *);
int64_t __wrap_et_g3c4_private_output_accept_text_v1(
    void *context, void *output, void *raw_candidate) {
  int64_t status = __real_et_g3c4_private_output_accept_text_v1(
      context, output, raw_candidate);
  if (status == 0 && shell_mutation != 0) {
    shell_bytevector *raw = (shell_bytevector *)raw_candidate;
    if ((shell_mutation & 1) != 0) raw->bytes[0] ^= 1u;
    if ((shell_mutation & 2) != 0 && shell_staging != NULL)
      shell_staging->bytes[0] ^= 1u;
    shell_mutation = 0;
  }
  return status;
}

#ifdef ET_G3C4_P2_G1_PUBLIC_RESULT_PRIVATE
static et_i64_tensor_borrow *result_test_borrow;
int64_t et_g3c4_result_test_value_v1(void *candidate, int64_t kind) {
  et_g3c4_result_tensor_internal *result =
      et_g3c4_admit_result(candidate, 0);
  et_i64_tensor_error error;
  et_i64_tensor_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *view = NULL;
  int64_t value = -1;
  if (result == NULL || result->transport.kind != (uint32_t)kind ||
      et_i64_tensor_borrow_begin_v1(result->tensor, &borrow, &error) != 0)
    return -1;
  if (et_i64_tensor_borrow_view_v1(borrow, &view, &error) == 0 &&
      view != NULL && view->rank == 1u && view->shape != NULL &&
      view->shape[0] == 1u && view->data != NULL &&
      view->byte_length == sizeof(int64_t) &&
      strcmp(view->dtype, "i64") == 0 &&
      strcmp(view->device, "cpu") == 0)
    memcpy(&value, view->data, sizeof(value));
  if (et_i64_tensor_borrow_end_v1(&borrow, &error) != 0) abort();
  return value;
}
int64_t et_g3c4_result_test_borrow_v1(void *candidate, int64_t active) {
  et_g3c4_result_tensor_internal *result =
      et_g3c4_admit_result(candidate, 0);
  et_i64_tensor_error error;
  if (result == NULL) return -1;
  if (active == 1 && result_test_borrow == NULL)
    return et_i64_tensor_borrow_begin_v1(
        result->tensor, &result_test_borrow, &error);
  if (active == 0 && result_test_borrow != NULL)
    return et_i64_tensor_borrow_end_v1(&result_test_borrow, &error);
  return -1;
}
#endif

#ifdef ET_G3C4_P2_G1_PUBLIC_RESULT_PRIVATE
int64_t et_g3c4_result_test_prior_view_device_v1(void *candidate) {
  et_g3c4_output_internal *output = et_g3c4_admit_output(candidate, 0);
  et_i64_tensor_error error;
  et_i64_tensor_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *view = NULL;
  if (output == NULL || output->ids == NULL ||
      et_i64_tensor_borrow_begin_v1(output->ids, &borrow, &error) != 0)
    return -1;
  if (et_i64_tensor_borrow_view_v1(borrow, &view, &error) != 0) abort();
  ((et_kernel_tensor_view_v1 *)view)->device = "prior-non-cpu";
  if (et_i64_tensor_borrow_end_v1(&borrow, &error) != 0) abort();
  return 0;
}
int64_t et_g3c4_result_test_output_shape_v1(
    void *candidate, int64_t malformed) {
  et_g3c4_output_internal *output = et_g3c4_admit_output(candidate, 0);
  if (output == NULL || output->ids == NULL ||
      (malformed != 0 && malformed != 1)) return -1;
  uint64_t *shape =
      (uint64_t *)et_i64_tensor_test_shape_storage_v1(output->ids);
  if (shape == NULL) return -1;
  shape[0] = malformed ? 2u : 1u;
  return 0;
}
#endif
