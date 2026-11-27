#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int64_t et_c2_private_model_sha256_v1(const void *, int64_t, void *);
int64_t et_c2_private_model_sha256_spans_experiment_v1(
    const void *, int64_t, const void *, int64_t, const void *, int64_t,
    void *);

typedef struct bytevector {
  int64_t length;
  uint8_t bytes[65536];
} bytevector;

#define CHECK(condition) do { if (!(condition)) { \
  fprintf(stderr, "C2 SHA span experiment failed at line %d\n", __LINE__); \
  return 1; \
} } while (0)

static void fill(bytevector *value, int64_t length, unsigned seed) {
  value->length = length;
  for (int64_t i = 0; i < length; ++i)
    value->bytes[i] = (uint8_t)(((uint64_t)i * 73u + seed) & 255u);
}

static int equal_to_whole(const bytevector *a, const bytevector *b,
                          const bytevector *c) {
  bytevector whole = {0, {0}}, direct = {32, {0}}, streamed = {32, {0}};
  const int64_t a_len = a ? a->length : 0;
  const int64_t b_len = b ? b->length : 0;
  const int64_t c_len = c ? c->length : 0;
  whole.length = a_len + b_len + c_len;
  if (whole.length > (int64_t)sizeof(whole.bytes)) return 0;
  if (a) memcpy(whole.bytes, a->bytes, (size_t)a_len);
  if (b) memcpy(whole.bytes + a_len, b->bytes, (size_t)b_len);
  if (c) memcpy(whole.bytes + a_len + b_len, c->bytes, (size_t)c_len);
  return et_c2_private_model_sha256_v1(&whole, whole.length, &direct) == 0 &&
         et_c2_private_model_sha256_spans_experiment_v1(
             a, a_len, b, b_len, c, c_len, &streamed) == 0 &&
         memcmp(direct.bytes, streamed.bytes, 32u) == 0;
}

int main(void) {
  static const char tensor_domain[] = "eshkol-checkpoint-tensor-v1\0";
  static const char container_domain[] = "eshkol-checkpoint-container-v1\0";
  static const int64_t lengths[] = {0, 1, 55, 56, 63, 64, 65, 127, 128,
                                    129, 4096, 64000};
  bytevector domain = {0, {0}}, record = {0, {0}}, payload = {0, {0}};
  bytevector output = {32, {0}}, before, payload_before;
  memcpy(domain.bytes, tensor_domain, sizeof(tensor_domain) - 1u);
  domain.length = (int64_t)sizeof(tensor_domain) - 1;
  fill(&record, 80, 23u);
  /* C1 hashes the record while its 32-byte digest slot is still zero. */
  memset(record.bytes + 48, 0, 32u);
  for (size_t i = 0; i < sizeof(lengths) / sizeof(lengths[0]); ++i) {
    fill(&payload, lengths[i], 47u);
    CHECK(equal_to_whole(&domain, &record, &payload));
  }
  memcpy(domain.bytes, container_domain, sizeof(container_domain) - 1u);
  domain.length = (int64_t)sizeof(container_domain) - 1;
  fill(&payload, 64000, 91u);
  CHECK(equal_to_whole(&domain, &payload, NULL));
  CHECK(equal_to_whole(NULL, NULL, NULL));
  {
    static const uint8_t abc_digest[32] = {
        0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
        0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
        0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
        0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad};
    domain.length = record.length = payload.length = 1;
    domain.bytes[0] = 'a'; record.bytes[0] = 'b'; payload.bytes[0] = 'c';
    CHECK(et_c2_private_model_sha256_spans_experiment_v1(
        &domain, 1, &record, 1, &payload, 1, &output) == 0);
    CHECK(memcmp(output.bytes, abc_digest, sizeof(abc_digest)) == 0);
  }
  domain.length = (int64_t)sizeof(container_domain) - 1;
  memcpy(domain.bytes, container_domain, sizeof(container_domain) - 1u);
  fill(&payload, 64000, 91u);

  before = output;
  CHECK(et_c2_private_model_sha256_spans_experiment_v1(
      &domain, domain.length + 1, &payload, payload.length,
      NULL, 0, &output) == 1);
  CHECK(memcmp(&output, &before, sizeof(output)) == 0);
  CHECK(et_c2_private_model_sha256_spans_experiment_v1(
      &domain, domain.length, &payload, payload.length,
      NULL, 1, &output) == 1);
  CHECK(memcmp(&output, &before, sizeof(output)) == 0);
  CHECK(et_c2_private_model_sha256_spans_experiment_v1(
      &domain, domain.length, &payload, -1, NULL, 0, &output) == 1);
  CHECK(memcmp(&output, &before, sizeof(output)) == 0);
  output.length = 31;
  before = output;
  CHECK(et_c2_private_model_sha256_spans_experiment_v1(
      &domain, domain.length, &payload, payload.length,
      NULL, 0, &output) == 1);
  CHECK(memcmp(&output, &before, sizeof(output)) == 0);
  output.length = 32;
  before = output;
  payload.length = 32;
  payload_before = payload;
  CHECK(et_c2_private_model_sha256_spans_experiment_v1(
      &domain, domain.length, &payload, payload.length,
      NULL, 0, &payload) == 1);
  CHECK(memcmp(&payload, &payload_before, sizeof(payload)) == 0);
  CHECK(memcmp(&output, &before, sizeof(output)) == 0);
  payload.length = 64000;
  memcpy(payload.bytes + 100, &output.length, sizeof(output.length));
  payload_before = payload;
  CHECK(et_c2_private_model_sha256_spans_experiment_v1(
      &domain, domain.length, &payload, payload.length,
      NULL, 0, payload.bytes + 100) == 1);
  CHECK(memcmp(&payload, &payload_before, sizeof(payload)) == 0);
  payload.length = INT64_MAX;
  before = output;
  CHECK(et_c2_private_model_sha256_spans_experiment_v1(
      &domain, domain.length, &payload, payload.length,
      NULL, 0, &output) == 1);
  CHECK(memcmp(&output, &before, sizeof(output)) == 0);
  puts("C2 SHA spans experiment: PASS");
  return 0;
}
