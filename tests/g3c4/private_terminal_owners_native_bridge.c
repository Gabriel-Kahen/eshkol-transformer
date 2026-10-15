/* Diagnostic-only bridge into the genuine kind-9 I1 borrow APIs. */
#define ET_G3C4_P2_G2_TERMINAL_IDS_CLONE_PRIVATE 1
#define ET_G3C4_P2_G2_TERMINAL_TEST_MAIN terminal_probe_predecessor_main
#include "test_p2_g2_native_terminal.c"

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
  if (clone == NULL || held_clone_ids != NULL) return -1;
  return et_i64_tensor_borrow_begin_v1(clone->ids, &held_clone_ids, &error);
}

int64_t et_g3c4_private_probe_clone_end_v1(void) {
  et_i64_tensor_error error;
  if (held_clone_ids == NULL) return -1;
  return et_i64_tensor_borrow_end_v1(&held_clone_ids, &error);
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
