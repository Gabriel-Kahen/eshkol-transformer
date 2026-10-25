/* C4 decoder adapter for pinned Eshkol bytevector length headers. */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#define ET_CPU4_C4_HISTORY_PRIVATE 1
#define ET_CPU4_C4_P3_WIRE_PRIVATE 1
#include "../src/eshkol_transformer/cpu4_c4_capture_private.h"

static void *cpu4_c4_decode_payload(void *header, int64_t expected,
                                     size_t alignment) {
  int64_t length;
  uintptr_t address = (uintptr_t)header;
  if (header == NULL ||
      address > UINTPTR_MAX - sizeof(length) - (uintptr_t)expected)
    return NULL;
  memcpy(&length, header, sizeof(length));
  if (length != expected ||
      (address + sizeof(length)) % alignment != 0u)
    return NULL;
  return (unsigned char *)header + sizeof(length);
}

int64_t et_cpu4_private_c4_p3_wire_decode_eshkol_v1(
    void *wire_header, int64_t wire_bytes,
    void *values_header, int64_t values_bytes,
    void *sha_header, void *fingerprint_header, void *record_header) {
  void *wire;
  void *values;
  void *sha;
  void *fingerprint;
  void *record;
  if (wire_bytes != 400 || values_bytes != 4768) return 1;
  wire = cpu4_c4_decode_payload(wire_header, 400, 1);
  values = cpu4_c4_decode_payload(values_header, 4768, 1);
  sha = cpu4_c4_decode_payload(sha_header, 32, 1);
  fingerprint = cpu4_c4_decode_payload(fingerprint_header, 96, 1);
  record = cpu4_c4_decode_payload(
      record_header, 224, _Alignof(et_cpu4_c4_capture_record_v1));
  if (wire == NULL || values == NULL || sha == NULL ||
      fingerprint == NULL || record == NULL) return 1;
  return et_cpu4_private_c4_p3_wire_decode_v1(
      wire, 400, values, 4768, sha, fingerprint, record);
}
