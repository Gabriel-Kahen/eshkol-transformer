/* Reuse the exact accepted prefix owner/rollback observer. This wrapper
 * forces a native terminal preflight rejection before its zero boundary. */
#include "p2_g2_t1_prefix_coordinator_native.c"

static int terminal_close_fault;

int64_t et_g3c4_p2g2_terminal_close_test_fault_v1(int64_t active) {
  if (active != 0 && active != 1) return -1;
  terminal_close_fault = (int)active;
  return 0;
}

int64_t __real_et_g3c4_private_p2g2_terminal_commit_v1(
    void *, void *, const void *, int64_t, const void *, int64_t);

int64_t __wrap_et_g3c4_private_p2g2_terminal_commit_v1(
    void *context, void *output, const void *first, int64_t first_bytes,
    const void *second, int64_t second_bytes) {
  if (terminal_close_fault) {
    et_g3c4_context_internal *native =
        et_g3c4_admit_active_call(context);
    if (native == NULL) return -1;
    if (native->p2g2_prefix_committed) {
      if (et_g3c4_p2g2_coordinator_test_prefix_v1(
              context, output, native->p2g2_prefix_token) != 0)
        return -1;
    } else if (et_g3c4_p2g2_coordinator_test_observe_v1(
                   context, output, 2, -1) != 0) {
      return -1;
    }
  }
  return __real_et_g3c4_private_p2g2_terminal_commit_v1(
      context, output, first, terminal_close_fault == 1 ? 8 : first_bytes,
      second, second_bytes);
}
