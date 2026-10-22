/* Diagnostic-only bridge into the genuine kind-9 I1 borrow APIs. */
#define ET_G3C4_P2_G2_TERMINAL_IDS_CLONE_PRIVATE 1
#define ET_G3C4_P2_G2_TERMINAL_TEST_MAIN terminal_probe_predecessor_main
#include "test_p2_g2_native_terminal.c"

#ifdef ET_G3C4_PROBE_TRACE
#include <execinfo.h>
#include <inttypes.h>
#include <stdlib.h>
extern int64_t et_m3_test_i64_counts_v1(int64_t field);
static void probe_trace(const char *operation, int64_t status) {
  fprintf(stderr, "G3-C4 PROBE %s status=%" PRId64
          " tensors=%" PRId64 " borrows=%" PRId64 "\n",
          operation, status, et_m3_test_i64_counts_v1(0),
          et_m3_test_i64_counts_v1(1));
  fflush(stderr);
}
extern void __real_exit(int status) __attribute__((noreturn));
extern void __real_abort(void) __attribute__((noreturn));
static void probe_stop_trace(const char *operation, int status) {
  void *frames[32];
  probe_trace(operation, status);
  backtrace_symbols_fd(frames, backtrace(frames, 32), STDERR_FILENO);
}
void __wrap_exit(int status) {
  probe_stop_trace("exit", status);
  __real_exit(status);
}
void __wrap_abort(void) {
  probe_stop_trace("abort", 134);
  __real_abort();
}
extern int64_t __real_et_g3c4_private_p2g2_terminal_ids_clone_release_v1(void *);
int64_t __wrap_et_g3c4_private_p2g2_terminal_ids_clone_release_v1(void *clone) {
  probe_trace("clone-release-before", 0);
  int64_t status =
      __real_et_g3c4_private_p2g2_terminal_ids_clone_release_v1(clone);
  probe_trace("clone-release-after", status);
  return status;
}
#else
#define probe_trace(operation, status) ((void)0)
#endif

static et_i64_tensor_borrow *held_clone_ids;
static et_i64_tensor_borrow *held_output_ids;

int64_t et_g3c4_private_probe_output_hold_v1(void *candidate) {
  et_g3c4_output_internal *output = et_g3c4_admit_output(candidate, 0);
  et_i64_tensor_error error;
  if (output == NULL || output->transport.state != ET_G3C4_OUTPUT_TERMINAL ||
      output->ids == NULL || held_output_ids != NULL)
    return -1;
  return et_i64_tensor_borrow_begin_v1(
      output->ids, &held_output_ids, &error);
}

int64_t et_g3c4_private_probe_output_end_v1(void) {
  et_i64_tensor_error error;
  if (held_output_ids == NULL) return -1;
  return et_i64_tensor_borrow_end_v1(&held_output_ids, &error);
}

int64_t et_g3c4_private_probe_output_live_v1(void *candidate) {
  et_g3c4_transport_header_internal *header;
  et_g3c4_output_internal *output;
  /* Admission requires an unborrowed I1 for terminal outputs. This probe
     intentionally observes the output while its I1 lease is held. */
  for (header = et_g3c4_transport_registry;
       header != NULL && (const void *)header != candidate;
       header = header->registry_next) {}
  if (header == NULL || header->kind != ET_G3C4_OUTPUT_KIND ||
      !et_g3c4_transport_record_valid(header))
    return -1;
  output = (et_g3c4_output_internal *)header;
  if (output->transport.state == ET_G3C4_OUTPUT_TERMINAL &&
      output->ids != NULL)
    return 0;
  if (output->transport.state == ET_G3C4_CONTEXT_DEAD &&
      output->ids == NULL)
    return 1;
  return -1;
}

int64_t et_g3c4_private_probe_clone_hold_v1(void *candidate) {
  et_g3c4_terminal_ids_clone_internal *clone =
      et_g3c4_terminal_ids_admit(candidate, 0);
  et_i64_tensor_error error;
  probe_trace("clone-hold-before", held_clone_ids != NULL);
  if (clone == NULL || held_clone_ids != NULL) return -1;
  int64_t status = et_i64_tensor_borrow_begin_v1(
      clone->ids, &held_clone_ids, &error);
  probe_trace("clone-hold-after", status);
  return status;
}

int64_t et_g3c4_private_probe_clone_end_v1(void) {
  et_i64_tensor_error error;
  if (held_clone_ids == NULL) return -1;
  probe_trace("clone-end-before", held_clone_ids != NULL);
  int64_t status = et_i64_tensor_borrow_end_v1(&held_clone_ids, &error);
  probe_trace("clone-end-after", status);
  return status;
}

int64_t et_g3c4_private_probe_clone_live_v1(void *candidate) {
  et_g3c4_terminal_ids_clone_internal *clone =
      et_g3c4_terminal_ids_admit(candidate, 1);
  if (clone == NULL) return -1;
  if (clone->transport.state == ET_G3C4_CONTEXT_LIVE && clone->ids != NULL)
    return 0;
  if (clone->transport.state == ET_G3C4_CONTEXT_DEAD && clone->ids == NULL)
    return 1;
  return -1;
}
