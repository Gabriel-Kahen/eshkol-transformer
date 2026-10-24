#ifndef ET_CPU4_C4_CAPTURE_PRIVATE_H
#define ET_CPU4_C4_CAPTURE_PRIVATE_H

#include <stddef.h>
#include <stdint.h>

#ifdef ET_CPU4_C4_HISTORY_PRIVATE

typedef struct et_cpu4_c4_capture et_cpu4_c4_capture;

typedef struct et_cpu4_c4_capture_record_v1 {
  uint32_t size;
  uint16_t major, minor;
  uint32_t flags, count, cache_capacity;
  uint32_t prompt_count, generated_count, manual_count;
  uint32_t generated_eos_mask, reserved0;
  uint64_t commit_epoch;
  int64_t ids[4];
  uint32_t origins[4];
  int64_t policy[6];
  int64_t rng[4];
  uint32_t parameter_count, values_bytes, state, reserved1;
  uint8_t digest[32];
} et_cpu4_c4_capture_record_v1;

_Static_assert(sizeof(et_cpu4_c4_capture_record_v1) == 224u, "capture record extent");
_Static_assert(offsetof(et_cpu4_c4_capture_record_v1, commit_epoch) == 40u, "capture epoch offset");
_Static_assert(offsetof(et_cpu4_c4_capture_record_v1, ids) == 48u, "capture ids offset");
_Static_assert(offsetof(et_cpu4_c4_capture_record_v1, origins) == 80u, "capture origins offset");
_Static_assert(offsetof(et_cpu4_c4_capture_record_v1, policy) == 96u, "capture policy offset");
_Static_assert(offsetof(et_cpu4_c4_capture_record_v1, rng) == 144u, "capture rng offset");
_Static_assert(offsetof(et_cpu4_c4_capture_record_v1, parameter_count) == 176u, "capture payload offset");
_Static_assert(offsetof(et_cpu4_c4_capture_record_v1, digest) == 192u, "capture digest offset");

et_cpu4_c4_capture *et_cpu4_private_c4_capture_create_v1(void *registered_generator);
int64_t et_cpu4_private_c4_capture_copy_record_v1(
    const et_cpu4_c4_capture *capture, et_cpu4_c4_capture_record_v1 *dst,
    uint64_t dst_bytes);
int64_t et_cpu4_private_c4_capture_copy_parameters_v1(
    const et_cpu4_c4_capture *capture, void *dst, uint64_t dst_bytes);
int64_t et_cpu4_private_c4_capture_release_v1(et_cpu4_c4_capture *capture);

#ifdef ET_CPU4_C4_P3_WIRE_PRIVATE
/* Source-private C4GRCV1 data codec. No registry or model authority is imported. */
int64_t et_cpu4_private_c4_p3_wire_encode_v1(
    const et_cpu4_c4_capture_record_v1 *record, const void *values,
    uint64_t values_bytes, const uint8_t c2_sha[32], const uint8_t t1_fp[96],
    void *wire, uint64_t wire_bytes);
int64_t et_cpu4_private_c4_p3_wire_decode_v1(
    const void *wire, uint64_t wire_bytes, const void *values,
    uint64_t values_bytes, const uint8_t expected_c2_sha[32],
    const uint8_t expected_t1_fp[96],
    et_cpu4_c4_capture_record_v1 *out_record);
#endif

#endif
#endif
