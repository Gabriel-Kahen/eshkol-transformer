/* Unrun compiled-gate fixture: inject a genuine held I1/A2 lease at the
 * first terminal commit, retain it across rejection, then let Eshkol's
 * private repair probe release it before exactly one native retry. */
#define ET_G3C4_P2_G2_TERMINAL_TEST_MAIN terminal_repair_predecessor_main
#include "test_p2_g2_native_terminal.c"
#undef ET_G3C4_P2_G2_TERMINAL_TEST_MAIN

static int terminal_repair_mode, terminal_repair_attempts;
static int terminal_repair_released, terminal_repair_first_nonzero;
static et_i64_tensor_borrow *terminal_repair_ids;
static et_a2_kv_cache_transaction_view *terminal_repair_view;

int64_t et_g3c4_terminal_repair_probe_failstop_mode_v1(void) {
  const char *mode = getenv("G3_TERMINAL_REPAIR_CASE");
  return mode != NULL && strcmp(mode, "held-false") == 0 ? 1 : 0;
}

int64_t et_g3c4_terminal_repair_probe_arm_v1(int64_t mode) {
  if ((mode != 1 && mode != 2) || terminal_repair_ids != NULL ||
      terminal_repair_view != NULL) return -1;
  terminal_repair_mode = (int)mode;
  terminal_repair_attempts = 0;
  terminal_repair_released = 0;
  terminal_repair_first_nonzero = 0;
  return 0;
}

int64_t __real_et_g3c4_private_p2g2_terminal_commit_v1(
    void *, void *, const void *, int64_t, const void *, int64_t);
int64_t __wrap_et_g3c4_private_p2g2_terminal_commit_v1(
    void *context_candidate, void *output_candidate,
    const void *first, int64_t first_bytes,
    const void *second, int64_t second_bytes) {
  et_g3c4_context_internal *context =
      (et_g3c4_context_internal *)context_candidate;
  et_g3c4_output_internal *output =
      (et_g3c4_output_internal *)output_candidate;
  et_i64_tensor_error ids_error;
  et_kernel_error cache_error;
  int64_t result;
  if (terminal_repair_mode == 0)
    return __real_et_g3c4_private_p2g2_terminal_commit_v1(
        context_candidate, output_candidate, first, first_bytes,
        second, second_bytes);
  terminal_repair_attempts++;
  if (terminal_repair_attempts == 1) {
    if (terminal_repair_mode == 1) {
      if (et_i64_tensor_borrow_begin_v1(
            output->ids, &terminal_repair_ids, &ids_error) != 0) abort();
    } else if (context->token_frame_transaction == NULL ||
               et_a2_kv_cache_transaction_view_begin_v1(
                 context->token_frame_transaction, 0u,
                 &terminal_repair_view, &cache_error) != 0) abort();
  } else if (terminal_repair_attempts != 2 ||
             terminal_repair_ids != NULL || terminal_repair_view != NULL ||
             !terminal_repair_released) abort();
  result = __real_et_g3c4_private_p2g2_terminal_commit_v1(
      context_candidate, output_candidate, first, first_bytes,
      second, second_bytes);
  if (terminal_repair_attempts == 1 && result != 0)
    terminal_repair_first_nonzero = 1;
  return result;
}

int64_t et_g3c4_terminal_repair_probe_pending_v1(
    void *context_candidate, void *output_candidate) {
  et_g3c4_context_internal *context =
      et_g3c4_admit_active_call(context_candidate);
  et_g3c4_output_internal *output =
      et_g3c4_admit_output(output_candidate, 0);
  return context != NULL && output != NULL &&
         terminal_repair_attempts == 1 && terminal_repair_first_nonzero &&
         !terminal_repair_released &&
         ((terminal_repair_mode == 1 && terminal_repair_ids != NULL) ||
          (terminal_repair_mode == 2 && terminal_repair_view != NULL)) &&
         output->transport.state == 0u && output->generated_length == 2 &&
         output->length == 0 && output->numeric_ready == 0u &&
         output->ids_copied == 0u && output->text_ready == 0u ? 0 : -1;
}

int64_t et_g3c4_terminal_repair_probe_release_v1(void) {
  et_i64_tensor_error ids_error;
  et_kernel_error cache_error;
  int64_t status = -1;
  if (terminal_repair_mode == 1 && terminal_repair_ids != NULL)
    status = et_i64_tensor_borrow_end_v1(&terminal_repair_ids, &ids_error);
  if (terminal_repair_mode == 2 && terminal_repair_view != NULL)
    status = et_a2_kv_cache_transaction_view_end_v1(
        &terminal_repair_view, &cache_error);
  if (status == 0) terminal_repair_released = 1;
  return status;
}

int64_t et_g3c4_terminal_repair_probe_done_v1(int64_t success) {
  int expected = success ? 2 : 1;
  int valid = terminal_repair_attempts == expected &&
              terminal_repair_first_nonzero && terminal_repair_released &&
              terminal_repair_ids == NULL && terminal_repair_view == NULL;
  terminal_repair_mode = 0;
  return valid ? 0 : -1;
}
