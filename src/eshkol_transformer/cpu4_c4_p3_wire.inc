/* Fixed, data-only C4GRCV1 P3 profile. Issuer admission belongs to Eshkol. */
enum { CPU4_P3_WIRE_BYTES = 400, CPU4_P3_FIELDS_BYTES = 192 };
static const uint8_t cpu4_p3_magic[8] = {'C','4','G','R','C','V','1',0};
static const uint8_t cpu4_p3_t1[] =
    "sha256:eshkol-byte-tokenizer-v1:aabb31f49216963582383a00fd2e85d7c20b658d5bef0e7945344ed6bfe50704";
_Static_assert(sizeof(cpu4_p3_t1) - 1u == 96u, "raw T1 fingerprint extent");

static uint16_t cpu4_p3_u16(const uint8_t *p) {
  return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8u));
}
static uint32_t cpu4_p3_u32(const uint8_t *p) {
  uint32_t v = 0u;
  for (unsigned i = 0u; i < 4u; ++i) v |= (uint32_t)p[i] << (8u * i);
  return v;
}
static uint64_t cpu4_p3_u64(const uint8_t *p) {
  uint64_t v = 0u;
  for (unsigned i = 0u; i < 8u; ++i) v |= (uint64_t)p[i] << (8u * i);
  return v;
}

static void cpu4_p3_record_encode(const et_cpu4_c4_capture_record_v1 *r,
                                  uint8_t out[224]) {
  memset(out, 0, 224u);
  cpu4_capture_le32(out, r->size);
  cpu4_capture_le16(out + 4u, r->major);
  cpu4_capture_le16(out + 6u, r->minor);
  cpu4_capture_le32(out + 8u, r->flags);
  cpu4_capture_le32(out + 12u, r->count);
  cpu4_capture_le32(out + 16u, r->cache_capacity);
  cpu4_capture_le32(out + 20u, r->prompt_count);
  cpu4_capture_le32(out + 24u, r->generated_count);
  cpu4_capture_le32(out + 28u, r->manual_count);
  cpu4_capture_le32(out + 32u, r->generated_eos_mask);
  cpu4_capture_le32(out + 36u, r->reserved0);
  cpu4_capture_le64(out + 40u, r->commit_epoch);
  for (unsigned i = 0u; i < 4u; ++i) {
    cpu4_capture_le64(out + 48u + 8u*i, (uint64_t)r->ids[i]);
    cpu4_capture_le32(out + 80u + 4u*i, r->origins[i]);
  }
  for (unsigned i = 0u; i < 6u; ++i)
    cpu4_capture_le64(out + 96u + 8u*i, (uint64_t)r->policy[i]);
  for (unsigned i = 0u; i < 4u; ++i)
    cpu4_capture_le64(out + 144u + 8u*i, (uint64_t)r->rng[i]);
  cpu4_capture_le32(out + 176u, r->parameter_count);
  cpu4_capture_le32(out + 180u, r->values_bytes);
  cpu4_capture_le32(out + 184u, r->state);
  cpu4_capture_le32(out + 188u, r->reserved1);
  memcpy(out + 192u, r->digest, 32u);
}

static int cpu4_p3_record_decode(const uint8_t in[224],
                                 et_cpu4_c4_capture_record_v1 *r) {
  memset(r, 0, sizeof(*r));
  r->size = cpu4_p3_u32(in);
  r->major = cpu4_p3_u16(in + 4u);
  r->minor = cpu4_p3_u16(in + 6u);
  r->flags = cpu4_p3_u32(in + 8u);
  r->count = cpu4_p3_u32(in + 12u);
  r->cache_capacity = cpu4_p3_u32(in + 16u);
  r->prompt_count = cpu4_p3_u32(in + 20u);
  r->generated_count = cpu4_p3_u32(in + 24u);
  r->manual_count = cpu4_p3_u32(in + 28u);
  r->generated_eos_mask = cpu4_p3_u32(in + 32u);
  r->reserved0 = cpu4_p3_u32(in + 36u);
  r->commit_epoch = cpu4_p3_u64(in + 40u);
  for (unsigned i = 0u; i < 4u; ++i) {
    const uint64_t id = cpu4_p3_u64(in + 48u + 8u*i);
    if (id > INT64_MAX) return 0;
    r->ids[i] = (int64_t)id;
    r->origins[i] = cpu4_p3_u32(in + 80u + 4u*i);
  }
  for (unsigned i = 0u; i < 6u; ++i) {
    const uint64_t value = cpu4_p3_u64(in + 96u + 8u*i);
    if (value > INT64_MAX) return 0;
    r->policy[i] = (int64_t)value;
  }
  for (unsigned i = 0u; i < 4u; ++i) {
    const uint64_t value = cpu4_p3_u64(in + 144u + 8u*i);
    if (value > INT64_MAX) return 0;
    r->rng[i] = (int64_t)value;
  }
  r->parameter_count = cpu4_p3_u32(in + 176u);
  r->values_bytes = cpu4_p3_u32(in + 180u);
  r->state = cpu4_p3_u32(in + 184u);
  r->reserved1 = cpu4_p3_u32(in + 188u);
  memcpy(r->digest, in + 192u, 32u);
  return 1;
}

static int cpu4_p3_profile(const et_cpu4_c4_capture_record_v1 *r) {
  return r->size == 224u && r->major == 1u && r->minor == 0u &&
      r->flags == 0u && r->count == 3u && r->cache_capacity == 4u &&
      r->prompt_count == 2u && r->generated_count == 1u &&
      r->manual_count == 0u && r->generated_eos_mask == 0u &&
      r->reserved0 == 0u && r->commit_epoch == 2u &&
      r->ids[0] == 3 && r->ids[1] == 197 && r->ids[2] == 3 && r->ids[3] == 0 &&
      r->origins[0] == 1u && r->origins[1] == 1u &&
      r->origins[2] == 2u && r->origins[3] == 0u &&
      r->policy[0] == 0 && r->policy[1] == INT64_C(0x3f800000) &&
      r->policy[2] == 256 && r->policy[3] == INT64_C(0x3f800000) &&
      r->policy[4] == 2 && (r->policy[5] == 41 || r->policy[5] == 255) &&
      r->rng[0] == 1 && r->rng[1] == 41 && r->rng[2] == 0 && r->rng[3] == 0 &&
      r->parameter_count == 14u && r->values_bytes == CPU4_CAPTURE_VALUES &&
      r->state == 1u && r->reserved1 == 0u;
}

/* Require all declared spans distinct, including input/input overlap. */
static int cpu4_p3_spans(const void *const ptr[5], const size_t size[5]) {
  for (unsigned i = 0u; i < 5u; ++i) {
    if (!et_g3c4_range_valid(ptr[i], size[i]) ||
        cpu4_capture_tls_alias(ptr[i], size[i], size[i])) return 0;
    for (unsigned j = 0u; j < i; ++j)
      if (et_g3c4_ranges_overlap(ptr[i], size[i], ptr[j], size[j])) return 0;
  }
  return 1;
}

int64_t et_cpu4_private_c4_p3_wire_encode_v1(
    const et_cpu4_c4_capture_record_v1 *record, const void *values,
    uint64_t values_bytes, const uint8_t c2_sha[32], const uint8_t t1_fp[96],
    void *wire, uint64_t wire_bytes) {
  const void *ptr[5] = {record, values, c2_sha, t1_fp, wire};
  const size_t size[5] = {224u, CPU4_CAPTURE_VALUES, 32u, 96u, CPU4_P3_WIRE_BYTES};
  cpu4_capture_payload payload;
  uint8_t encoded[CPU4_P3_WIRE_BYTES];
  uint8_t digest[32];
  cpu4_sha256_state sha;
  if (values_bytes != CPU4_CAPTURE_VALUES || wire_bytes != CPU4_P3_WIRE_BYTES ||
      !cpu4_p3_spans(ptr, size)) return ET_G3C4_INVALID_ARGUMENT;
  et_g3c4_error_reset_internal();
  if ((uintptr_t)record % _Alignof(et_cpu4_c4_capture_record_v1) != 0u ||
      cpu4_capture_copy_alias(wire, CPU4_P3_WIRE_BYTES) != 0)
    return et_g3c4_fail(ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_ALIAS);
  memcpy(&payload.record, record, sizeof(*record));
  memcpy(payload.values, values, CPU4_CAPTURE_VALUES);
  if (!cpu4_p3_profile(&payload.record) ||
      !cpu4_capture_record_valid(&payload, payload.record.digest) ||
      memcmp(t1_fp, cpu4_p3_t1, 96u) != 0)
    return et_g3c4_fail(ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
  memset(encoded, 0, sizeof(encoded));
  memcpy(encoded, cpu4_p3_magic, 8u);
  cpu4_capture_le32(encoded + 8u, CPU4_P3_WIRE_BYTES);
  cpu4_capture_le32(encoded + 12u, 1u);
  cpu4_p3_record_encode(&payload.record, encoded + 16u);
  memcpy(encoded + 240u, c2_sha, 32u);
  memcpy(encoded + 272u, t1_fp, 96u);
  cpu4_sha256_init(&sha);
  cpu4_sha256_update(&sha, encoded, 368u);
  cpu4_sha256_final(&sha, digest);
  memcpy(encoded + 368u, digest, 32u);
  memcpy(wire, encoded, sizeof(encoded));
  return 0;
}

int64_t et_cpu4_private_c4_p3_wire_decode_v1(
    const void *wire, uint64_t wire_bytes, const void *values,
    uint64_t values_bytes, const uint8_t expected_c2_sha[32],
    const uint8_t expected_t1_fp[96],
    et_cpu4_c4_capture_record_v1 *out_record) {
  const void *ptr[5] = {wire, values, expected_c2_sha, expected_t1_fp, out_record};
  const size_t size[5] = {CPU4_P3_WIRE_BYTES, CPU4_CAPTURE_VALUES, 32u, 96u, 224u};
  const uint8_t *bytes = wire;
  cpu4_capture_payload payload;
  uint8_t digest[32];
  cpu4_sha256_state sha;
  if (wire_bytes != CPU4_P3_WIRE_BYTES || values_bytes != CPU4_CAPTURE_VALUES ||
      !cpu4_p3_spans(ptr, size)) return ET_G3C4_INVALID_ARGUMENT;
  et_g3c4_error_reset_internal();
  if ((uintptr_t)out_record % _Alignof(et_cpu4_c4_capture_record_v1) != 0u ||
      cpu4_capture_copy_alias(out_record, sizeof(*out_record)) != 0)
    return et_g3c4_fail(ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_ALIAS);
  if (memcmp(bytes, cpu4_p3_magic, 8u) != 0 ||
      cpu4_p3_u32(bytes + 8u) != CPU4_P3_WIRE_BYTES ||
      cpu4_p3_u32(bytes + 12u) != 1u ||
      memcmp(bytes + 240u, expected_c2_sha, 32u) != 0 ||
      memcmp(bytes + 272u, expected_t1_fp, 96u) != 0 ||
      memcmp(expected_t1_fp, cpu4_p3_t1, 96u) != 0)
    return et_g3c4_fail(ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
  cpu4_sha256_init(&sha);
  cpu4_sha256_update(&sha, bytes, 368u);
  cpu4_sha256_final(&sha, digest);
  if (memcmp(bytes + 368u, digest, 32u) != 0 ||
      !cpu4_p3_record_decode(bytes + 16u, &payload.record) ||
      !cpu4_p3_profile(&payload.record))
    return et_g3c4_fail(ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
  memcpy(payload.values, values, CPU4_CAPTURE_VALUES);
  if (!cpu4_capture_record_valid(&payload, payload.record.digest))
    return et_g3c4_fail(ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
  memcpy(out_record, &payload.record, sizeof(*out_record));
  return 0;
}
