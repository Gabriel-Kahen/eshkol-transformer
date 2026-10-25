/* The pinned Eshkol bytevector pointer names its i64 length header. Keep
 * this ABI conversion next to the raw capture/codec owner, which retains all
 * registry and P3 profile checks. This feature is private to the P3 fixture. */
static uint8_t *cpu4_p3_boxed_payload(const void *boxed, size_t expected) {
  int64_t declared;
  if (expected > SIZE_MAX - sizeof(declared) ||
      !et_g3c4_range_valid(boxed, sizeof(declared) + expected) ||
      cpu4_capture_tls_alias(boxed, sizeof(declared) + expected,
                             sizeof(declared) + expected)) return NULL;
  memcpy(&declared, boxed, sizeof(declared));
  if (declared < 0 || (uint64_t)declared != expected) return NULL;
  return (uint8_t *)boxed + sizeof(declared);
}

int64_t et_cpu4_private_c4_capture_copy_record_eshkol_v1(
    const et_cpu4_c4_capture *capture, void *boxed_record) {
  uint8_t *payload;
  et_g3c4_error_reset_internal();
  payload = cpu4_p3_boxed_payload(boxed_record, 224u);
  if (payload == NULL)
    return et_g3c4_fail(ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
  if (cpu4_capture_copy_alias(boxed_record, 232u) != 0)
    return et_g3c4_error_state.category;
  return et_cpu4_private_c4_capture_copy_record_v1(
      capture, (et_cpu4_c4_capture_record_v1 *)payload, 224u);
}

int64_t et_cpu4_private_c4_capture_copy_parameters_eshkol_v1(
    const et_cpu4_c4_capture *capture, void *boxed_values) {
  uint8_t *payload;
  et_g3c4_error_reset_internal();
  payload = cpu4_p3_boxed_payload(boxed_values, CPU4_CAPTURE_VALUES);
  if (payload == NULL)
    return et_g3c4_fail(ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
  if (cpu4_capture_copy_alias(boxed_values, 8u + CPU4_CAPTURE_VALUES) != 0)
    return et_g3c4_error_state.category;
  return et_cpu4_private_c4_capture_copy_parameters_v1(
      capture, payload, CPU4_CAPTURE_VALUES);
}

int64_t et_cpu4_private_c4_p3_wire_encode_eshkol_v1(
    const void *boxed_record, const void *boxed_values,
    const void *boxed_c2_sha, const void *boxed_t1_fp, void *boxed_wire) {
  const void *boxed[5] = {boxed_record, boxed_values, boxed_c2_sha,
                          boxed_t1_fp, boxed_wire};
  const size_t size[5] = {224u, CPU4_CAPTURE_VALUES, 32u, 96u,
                          CPU4_P3_WIRE_BYTES};
  uint8_t *payload[5];
  et_g3c4_error_reset_internal();
  for (unsigned i = 0u; i < 5u; ++i) {
    payload[i] = cpu4_p3_boxed_payload(boxed[i], size[i]);
    if (payload[i] == NULL)
      return et_g3c4_fail(ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
    for (unsigned j = 0u; j < i; ++j)
      if (et_g3c4_ranges_overlap(boxed[i], size[i] + 8u,
                                 boxed[j], size[j] + 8u))
        return et_g3c4_fail(ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_ALIAS);
  }
  if (cpu4_capture_copy_alias(boxed_wire, CPU4_P3_WIRE_BYTES + 8u) != 0)
    return et_g3c4_error_state.category;
  return et_cpu4_private_c4_p3_wire_encode_v1(
      (const et_cpu4_c4_capture_record_v1 *)payload[0], payload[1],
      CPU4_CAPTURE_VALUES, payload[2], payload[3], payload[4],
      CPU4_P3_WIRE_BYTES);
}
