/* Pinned Eshkol bytevectors enter C at their i64 length header. */
#include <stdint.h>
#include <stddef.h>
#include <string.h>

extern int64_t et_cpu4_private_c4_p3_resume_prefix_v1(
    void *, void *, const int64_t[3], const int64_t[4], float[256]);

static void *cpu4_c4_p3_payload(void *header, int64_t expected,
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

int64_t et_cpu4_private_c4_p3_resume_eshkol_v1(
    void *context, void *pending_output, void *ids_header,
    void *rng_header, void *logits_header) {
  int64_t *ids = cpu4_c4_p3_payload(ids_header, 24, _Alignof(int64_t));
  int64_t *rng = cpu4_c4_p3_payload(rng_header, 32, _Alignof(int64_t));
  float *logits = cpu4_c4_p3_payload(logits_header, 1024, _Alignof(float));
  if (ids == NULL || rng == NULL || logits == NULL) return 1;
  return et_cpu4_private_c4_p3_resume_prefix_v1(
      context, pending_output, ids, rng, logits);
}
