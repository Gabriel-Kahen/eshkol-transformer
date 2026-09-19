#include "c2_checkpoint_save_bridge.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct bytevector {
  int64_t length;
  uint8_t bytes[];
} bytevector;

static int checks;
#define CHECK(condition)                                                        \
  do {                                                                          \
    ++checks;                                                                    \
    if (!(condition)) {                                                          \
      fprintf(stderr, "save bridge check failed at %s:%d\n", __FILE__,         \
              __LINE__);                                                         \
      return 1;                                                                  \
    }                                                                            \
  } while (0)

static uint32_t u32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8u) |
         ((uint32_t)p[2] << 16u) | ((uint32_t)p[3] << 24u);
}

static uint64_t u64(const uint8_t *p) {
  uint64_t value = 0u;
  unsigned i;
  for (i = 0u; i < 8u; ++i) value |= (uint64_t)p[i] << (8u * i);
  return value;
}

static bytevector *make_bytes(const uint8_t *source, size_t length) {
  bytevector *value;
  if (length > (size_t)INT64_MAX || length > SIZE_MAX - sizeof(*value))
    return NULL;
  value = (bytevector *)malloc(sizeof(*value) + length);
  if (value == NULL) return NULL;
  value->length = (int64_t)length;
  if (length != 0u) memcpy(value->bytes, source, length);
  return value;
}

static bytevector *zero_bytes(size_t length) {
  bytevector *value = make_bytes(NULL, 0u);
  if (value == NULL) return NULL;
  free(value);
  value = (bytevector *)malloc(sizeof(*value) + length);
  if (value == NULL) return NULL;
  value->length = (int64_t)length;
  memset(value->bytes, 0, length);
  return value;
}

static int measure(bytevector **parts, int64_t file, int64_t metadata,
                   int64_t tensor, int64_t tensors, int64_t profile,
                   bytevector *result) {
  return (int)et_c2_private_checkpoint_save_measure_v1(
      parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6],
      2, 0, 1729, 0, 0, file, metadata, tensor, tensors, profile, result);
}

int main(int argc, char **argv) {
  FILE *stream;
  uint8_t *file;
  long end;
  size_t length, x1_at, current_at, epoch_at, optimizer_at, model_at;
  size_t tokenizer_n, x1_n, current_n, epoch_n, optimizer_n, model_n, moments_n;
  uint64_t metadata_n;
  bytevector *parts[7] = {NULL};
  bytevector *result = NULL, *destination = NULL;
  size_t i;
  int status;
  if (argc != 2) return 2;
  stream = fopen(argv[1], "rb");
  CHECK(stream != NULL);
  CHECK(fseek(stream, 0, SEEK_END) == 0);
  end = ftell(stream);
  CHECK(end > 0 && (uintmax_t)end <= SIZE_MAX);
  length = (size_t)end;
  CHECK(fseek(stream, 0, SEEK_SET) == 0);
  file = (uint8_t *)malloc(length);
  CHECK(file != NULL);
  CHECK(fread(file, 1u, length, stream) == length);
  CHECK(fclose(stream) == 0);

  tokenizer_n = u32(file + 136u);
  x1_n = (size_t)u64(file + 120u);
  current_n = u32(file + 128u);
  epoch_n = u32(file + 132u);
  optimizer_n = (size_t)u64(file + 104u);
  model_n = (size_t)u64(file + 96u);
  moments_n = (size_t)u64(file + 112u);
  x1_at = 256u + tokenizer_n + u32(file + 140u) + u32(file + 144u) +
          u32(file + 148u) + u32(file + 152u);
  current_at = x1_at + x1_n;
  epoch_at = current_at + current_n;
  optimizer_at = epoch_at + epoch_n;
  model_at = (size_t)u64(file + 64u);
  CHECK(model_at + model_n + moments_n + 32u == length);
  parts[0] = make_bytes(file + 256u, tokenizer_n);
  parts[1] = make_bytes(file + x1_at, x1_n);
  parts[2] = make_bytes(file + current_at, current_n);
  parts[3] = make_bytes(file + epoch_at, epoch_n);
  parts[4] = make_bytes(file + model_at, model_n);
  parts[5] = make_bytes(file + optimizer_at, optimizer_n);
  parts[6] = make_bytes(file + model_at + model_n, moments_n);
  for (i = 0u; i < 7u; ++i) CHECK(parts[i] != NULL);
  result = zero_bytes((size_t)ET_C2_CHECKPOINT_SAVE_RESULT_BYTES);
  CHECK(result != NULL);

  status = measure(parts, INT64_C(1099511627776), INT64_C(268435456),
                   INT64_C(274877906944), 6826, 0, result);
  CHECK(status == 0);
  CHECK(u64(result->bytes) == ET_C2_CHECKPOINT_SAVE_RESULT_MAGIC);
  CHECK(u64(result->bytes + 8u) == length);
  metadata_n = u64(file + 56u) + u64(parts[4]->bytes + 56u);

  memset(result->bytes, 0, (size_t)result->length);
  CHECK(measure(parts, (int64_t)length - 1, INT64_C(268435456),
                INT64_C(274877906944), 6826, 0, result) != 0);
  CHECK(u64(result->bytes) == ET_C2_CHECKPOINT_SAVE_PHASE_MEASURE);
  CHECK(u64(result->bytes + 16u) == 7u);
  memset(result->bytes, 0, (size_t)result->length);
  CHECK(measure(parts, INT64_C(1099511627776), (int64_t)metadata_n - 1,
                INT64_C(274877906944), 6826, 0, result) != 0);
  CHECK(u64(result->bytes + 16u) == 7u);
  memset(result->bytes, 0, (size_t)result->length);
  CHECK(measure(parts, INT64_C(1099511627776), INT64_C(268435456), 7, 6826,
                0, result) != 0);
  CHECK(u64(result->bytes + 16u) == 7u);
  memset(result->bytes, 0, (size_t)result->length);
  CHECK(measure(parts, INT64_C(1099511627776), INT64_C(268435456),
                INT64_C(274877906944), 2, 0, result) != 0);
  CHECK(u64(result->bytes + 16u) == 7u);
  CHECK(measure(parts, INT64_C(1099511627776), INT64_C(268435456),
                INT64_C(274877906944), 6826, 1, result) == 0);

  destination = zero_bytes(length);
  CHECK(destination != NULL);
  memset(result->bytes, 0, (size_t)result->length);
  status = (int)et_c2_private_checkpoint_save_encode_validate_v1(
      parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6],
      2, 0, 1729, 0, 0, INT64_C(1099511627776), INT64_C(268435456),
      INT64_C(274877906944), 6826, 0, destination, result);
  CHECK(status == 0);
  CHECK(u64(result->bytes) == ET_C2_CHECKPOINT_SAVE_RESULT_MAGIC);
  CHECK(memcmp(destination->bytes, file, length) == 0);

  et_c2_checkpoint_save_test_fail_stage_v1(ET_C2_CHECKPOINT_SAVE_PHASE_ENCODE);
  memset(destination->bytes, 0xa5, length);
  CHECK(et_c2_private_checkpoint_save_encode_validate_v1(
            parts[0], parts[1], parts[2], parts[3], parts[4], parts[5],
            parts[6], 2, 0, 1729, 0, 0, INT64_C(1099511627776),
            INT64_C(268435456), INT64_C(274877906944), 6826, 0, destination,
            result) != 0);
  CHECK(destination->bytes[0] == 0xa5u &&
        destination->bytes[length - 1u] == 0xa5u);
  et_c2_checkpoint_save_test_fail_stage_v1(ET_C2_CHECKPOINT_SAVE_PHASE_PARSE);
  CHECK(et_c2_private_checkpoint_save_encode_validate_v1(
            parts[0], parts[1], parts[2], parts[3], parts[4], parts[5],
            parts[6], 2, 0, 1729, 0, 0, INT64_C(1099511627776),
            INT64_C(268435456), INT64_C(274877906944), 6826, 0, destination,
            result) != 0);
  CHECK(u64(result->bytes) == ET_C2_CHECKPOINT_SAVE_PHASE_PARSE);
  et_c2_checkpoint_save_test_fail_stage_v1(
      ET_C2_CHECKPOINT_SAVE_PHASE_POSTCONDITION);
  CHECK(et_c2_private_checkpoint_save_encode_validate_v1(
            parts[0], parts[1], parts[2], parts[3], parts[4], parts[5],
            parts[6], 2, 0, 1729, 0, 0, INT64_C(1099511627776),
            INT64_C(268435456), INT64_C(274877906944), 6826, 0, destination,
            result) != 0);
  CHECK(u64(result->bytes) == ET_C2_CHECKPOINT_SAVE_PHASE_POSTCONDITION);
  et_c2_checkpoint_save_test_fail_stage_v1(0);

  CHECK(et_c2_private_checkpoint_save_measure_v1(
            parts[0], parts[1], parts[2], parts[3], parts[4], parts[5],
            parts[6], 2, 0, 1729, 0, 0, INT64_C(1099511627776),
            INT64_C(268435456), INT64_C(274877906944), 6826, 0,
            parts[0]) != 0);
  CHECK(memcmp(parts[0]->bytes, file + 256u, tokenizer_n) == 0);
  parts[0]->bytes[0] = (uint8_t)'@';
  CHECK(measure(parts, INT64_C(1099511627776), INT64_C(268435456),
                INT64_C(274877906944), 6826, 0, result) != 0);
  CHECK(u64(result->bytes) == ET_C2_CHECKPOINT_SAVE_PHASE_MEASURE);

  for (i = 0u; i < 7u; ++i) free(parts[i]);
  free(result);
  free(destination);
  free(file);
  printf("C2 checkpoint save bridge PASS: %d checks\n", checks);
  return 0;
}
