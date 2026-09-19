#define _GNU_SOURCE

#include "c2_checkpoint_load_bridge.h"
#include "c2_checkpoint_core.h"
#include "c2_checkpoint_reader.h"
#include "checkpoint_io.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct bv {
  int64_t length;
  uint8_t bytes[];
} bv;

enum { CANARY_BYTES = 16 };
static const uint8_t canary_value = UINT8_C(0xd3);
static unsigned checks;

#define CHECK(x)                                                               \
  do {                                                                         \
    ++checks;                                                                  \
    if (!(x)) {                                                                \
      fprintf(stderr, "LOAD BRIDGE FAIL line %d: %s\n", __LINE__, #x);       \
      exit(1);                                                                 \
    }                                                                          \
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

/* Every ordinary test bytevector is surrounded by canaries. The capacity is
 * hidden before the prefix so declared-length mutation cannot affect cleanup
 * or verification. */
static bv *make_bv(size_t length, uint8_t fill) {
  const size_t total = sizeof(size_t) + CANARY_BYTES + sizeof(bv) + length +
                       CANARY_BYTES;
  uint8_t *raw = (uint8_t *)malloc(total);
  bv *value;
  CHECK(raw != NULL);
  memcpy(raw, &length, sizeof(length));
  memset(raw + sizeof(length), canary_value, CANARY_BYTES);
  value = (bv *)(void *)(raw + sizeof(length) + CANARY_BYTES);
  value->length = (int64_t)length;
  memset(value->bytes, fill, length);
  memset(value->bytes + length, canary_value, CANARY_BYTES);
  return value;
}

static size_t bv_capacity(const bv *value) {
  size_t length;
  const uint8_t *raw = (const uint8_t *)(const void *)value - CANARY_BYTES -
                       sizeof(size_t);
  memcpy(&length, raw, sizeof(length));
  return length;
}

static int bv_canaries(const bv *value) {
  const size_t length = bv_capacity(value);
  const uint8_t *prefix = (const uint8_t *)(const void *)value - CANARY_BYTES;
  const uint8_t *suffix = value->bytes + length;
  size_t i;
  for (i = 0u; i < CANARY_BYTES; ++i)
    if (prefix[i] != canary_value || suffix[i] != canary_value) return 0;
  return 1;
}

static int bv_fill_unchanged(const bv *value, uint8_t fill) {
  const size_t length = bv_capacity(value);
  size_t i;
  for (i = 0u; i < length; ++i)
    if (value->bytes[i] != fill) return 0;
  return bv_canaries(value);
}

static void free_bv(bv *value) {
  uint8_t *raw;
  CHECK(bv_canaries(value));
  raw = (uint8_t *)(void *)value - CANARY_BYTES - sizeof(size_t);
  free(raw);
}

static bv *clone_bv(const bv *source) {
  const size_t length = bv_capacity(source);
  bv *result = make_bv(length, 0u);
  result->length = source->length;
  memcpy(result->bytes, source->bytes, length);
  return result;
}

static bv *path_bv(const char *path) {
  const size_t length = strlen(path);
  bv *value = make_bv(length, 0u);
  memcpy(value->bytes, path, length);
  return value;
}

static bv *nested_bv(bv *container, size_t offset, size_t length) {
  bv *value;
  CHECK(offset <= bv_capacity(container));
  CHECK(sizeof(bv) + length <= bv_capacity(container) - offset);
  value = (bv *)(void *)(container->bytes + offset);
  value->length = (int64_t)length;
  return value;
}

static int64_t measure(const bv *path, bv *result) {
  return et_c2_private_checkpoint_load_measure_v1(
      path, INT64_C(1048576), INT64_C(1048576), INT64_C(1048576),
      INT64_C(16), 0, result);
}

static int64_t stage(const bv *path, const bv *measurement, bv **out,
                     bv *result) {
  return et_c2_private_checkpoint_load_stage_v1(
      path, INT64_C(1048576), INT64_C(1048576), INT64_C(1048576),
      INT64_C(16), 0, measurement, out[0], out[1], out[2], out[3], out[4],
      out[5], out[6], out[7], result);
}

static size_t output_length(const bv *measurement, size_t index) {
  static const size_t offsets[8] = {16u, 0u, 24u, 32u,
                                    40u, 48u, 56u, 64u};
  return index == 1u ? 93u :
         (size_t)u64(measurement->bytes + offsets[index]);
}

static void allocate_outputs(const bv *measurement, bv **out, uint8_t fill) {
  size_t i;
  for (i = 0u; i < 8u; ++i)
    out[i] = make_bv(output_length(measurement, i), fill);
}

static int outputs_unchanged(bv **out, uint8_t fill) {
  size_t i;
  for (i = 0u; i < 8u; ++i)
    if (!bv_fill_unchanged(out[i], fill)) return 0;
  return 1;
}

static int output_canaries(bv **out) {
  size_t i;
  for (i = 0u; i < 8u; ++i)
    if (!bv_canaries(out[i])) return 0;
  return 1;
}

static void free_outputs(bv **out) {
  size_t i;
  for (i = 0u; i < 8u; ++i) free_bv(out[i]);
}

static uint8_t *read_file(const char *path, size_t *length) {
  FILE *file = fopen(path, "rb");
  uint8_t *data;
  long end;
  CHECK(file != NULL);
  CHECK(fseek(file, 0L, SEEK_END) == 0);
  end = ftell(file);
  CHECK(end >= 0);
  CHECK(fseek(file, 0L, SEEK_SET) == 0);
  *length = (size_t)end;
  data = (uint8_t *)malloc(*length == 0u ? 1u : *length);
  CHECK(data != NULL);
  CHECK(fread(data, 1u, *length, file) == *length);
  CHECK(fclose(file) == 0);
  return data;
}

static void check_exact_outputs(const uint8_t *file, size_t file_length,
                                bv **out) {
  const size_t tokenizer = 256u;
  const size_t config = tokenizer + (size_t)u32(file + 136u);
  const size_t x1 = config + 93u + 19u + 57u + 71u;
  const size_t current = x1 + (size_t)u64(file + 120u);
  const size_t epoch = current + (size_t)u32(file + 128u);
  const size_t optimizer = epoch + (size_t)u32(file + 132u);
  const size_t model = (size_t)u64(file + 64u);
  const size_t moments = model + (size_t)u64(file + 96u);
  const size_t offsets[8] = {tokenizer, config, x1, current, epoch,
                             model, optimizer, moments};
  size_t i;
  for (i = 0u; i < 8u; ++i) {
    CHECK(offsets[i] <= file_length);
    CHECK(bv_capacity(out[i]) <= file_length - offsets[i]);
    CHECK(memcmp(out[i]->bytes, file + offsets[i], bv_capacity(out[i])) == 0);
  }
}

static void replace_file(const char *source, const char *destination) {
  FILE *in = fopen(source, "rb");
  FILE *out = fopen(destination, "wb");
  uint8_t buffer[4096];
  size_t amount;
  CHECK(in != NULL && out != NULL);
  while ((amount = fread(buffer, 1u, sizeof(buffer), in)) != 0u)
    CHECK(fwrite(buffer, 1u, amount, out) == amount);
  CHECK(!ferror(in));
  CHECK(fclose(in) == 0);
  CHECK(fclose(out) == 0);
}

static void check_atomic_failure(const bv *path, const bv *measurement,
                                 bv **out, bv *result, uint8_t fill) {
  uint8_t result_snapshot[ET_C2_CHECKPOINT_LOAD_RESULT_BYTES];
  memcpy(result_snapshot, result->bytes, sizeof(result_snapshot));
  et_c2_checkpoint_load_test_reset_v1();
  CHECK(stage(path, measurement, out, result) == 1);
  CHECK(et_c2_checkpoint_load_test_copy_count_v1() == 0u);
  CHECK(outputs_unchanged(out, fill));
  CHECK(memcmp(result_snapshot, result->bytes, sizeof(result_snapshot)) == 0);
}

static void check_io_failure(const bv *path, const bv *measurement,
                             bv **out, bv *result, uint8_t fill,
                             int64_t expected) {
  int64_t actual;
  et_c2_checkpoint_load_test_reset_v1();
  actual = stage(path, measurement, out, result);
  if (actual != expected)
    fprintf(stderr, "LOAD BRIDGE unexpected status: got %lld expected %lld\n",
            (long long)actual, (long long)expected);
  CHECK(actual == expected);
  CHECK(et_c2_checkpoint_load_test_copy_count_v1() == 0u);
  CHECK(outputs_unchanged(out, fill));
  CHECK(et_c2_checkpoint_core_test_live_images_v1() == 0u);
  CHECK(et_c2_checkpoint_reader_test_live_count_v1() == 0u);
  CHECK(et_c2_checkpoint_reader_test_fd_live_count_v1() == 0u);
}

int main(int argc, char **argv) {
  bv *path, *measurement, *result, *out[8];
  bv *hostile_path, *hostile_i64_path;
  uint8_t *valid_file;
  size_t valid_file_length;
  char mutable_path[4096];
  int64_t status;
  size_t i;
  CHECK(argc == 6);
  path = path_bv(argv[1]);
  hostile_path = path_bv(argv[2]);
  hostile_i64_path = path_bv(argv[3]);
  valid_file = read_file(argv[1], &valid_file_length);
  measurement = make_bv((size_t)ET_C2_CHECKPOINT_LOAD_RESULT_BYTES, 0xa5u);
  result = make_bv((size_t)ET_C2_CHECKPOINT_LOAD_RESULT_BYTES, 0xa5u);

  CHECK(measure(path, measurement) == 0);
  CHECK(u64(measurement->bytes) == ET_C2_CHECKPOINT_LOAD_RESULT_MAGIC);
  CHECK(u64(measurement->bytes + 16u) == 96u);
  CHECK(u64(measurement->bytes + 120u) == 1u);
  allocate_outputs(measurement, out, 0xa5u);
  et_c2_checkpoint_load_test_reset_v1();
  CHECK(stage(path, measurement, out, result) == 0);
  CHECK(et_c2_checkpoint_load_test_copy_count_v1() == 8u);
  CHECK(u64(result->bytes) == ET_C2_CHECKPOINT_LOAD_RESULT_MAGIC);
  check_exact_outputs(valid_file, valid_file_length, out);
  CHECK(output_canaries(out));
  CHECK(bv_canaries(path) && bv_canaries(measurement) && bv_canaries(result));
  free_outputs(out);

  allocate_outputs(measurement, out, 0x6du);
  et_c2_checkpoint_load_test_reset_v1();
  et_c2_checkpoint_load_test_fail_after_validate_v1(1);
  CHECK(stage(path, measurement, out, result) == 11);
  CHECK(et_c2_checkpoint_load_test_copy_count_v1() == 0u);
  CHECK(outputs_unchanged(out, 0x6du));
  free_outputs(out);

  /* Every output rejects both an under- and over-declared length before the
   * first copy, preserving all destination bytes and canaries. */
  for (i = 0u; i < 8u; ++i) {
    int delta;
    for (delta = -1; delta <= 1; delta += 2) {
      allocate_outputs(measurement, out, 0x3cu);
      out[i]->length += delta;
      check_atomic_failure(path, measurement, out, result, 0x3cu);
      out[i]->length -= delta;
      free_outputs(out);
    }
  }

  /* NULL output admission is also pre-I/O and preserves every destination
   * and the valid result object. */
  for (i = 0u; i < 8u; ++i) {
    bv *saved;
    uint8_t result_snapshot[ET_C2_CHECKPOINT_LOAD_RESULT_BYTES];
    size_t j;
    allocate_outputs(measurement, out, 0x3du);
    saved = out[i];
    out[i] = NULL;
    memcpy(result_snapshot, result->bytes, sizeof(result_snapshot));
    et_c2_checkpoint_load_test_reset_v1();
    CHECK(stage(path, measurement, out, result) == 1);
    CHECK(et_c2_checkpoint_load_test_copy_count_v1() == 0u);
    CHECK(memcmp(result_snapshot, result->bytes, sizeof(result_snapshot)) == 0);
    CHECK(bv_fill_unchanged(saved, 0x3du));
    for (j = 0u; j < 8u; ++j)
      if (j != i) CHECK(bv_fill_unchanged(out[j], 0x3du));
    out[i] = saved;
    free_outputs(out);
  }

  allocate_outputs(measurement, out, 0x7eu);
  free_bv(out[1]);
  out[1] = out[0];
  et_c2_checkpoint_load_test_reset_v1();
  CHECK(stage(path, measurement, out, result) == 1);
  CHECK(et_c2_checkpoint_load_test_copy_count_v1() == 0u);
  CHECK(bv_fill_unchanged(out[0], 0x7eu));
  for (i = 2u; i < 8u; ++i) free_bv(out[i]);
  free_bv(out[0]);

  allocate_outputs(measurement, out, 0x4au);
  {
    bv *container = make_bv(256u, 0x4au);
    uint8_t snapshot[256];
    free_bv(out[0]);
    free_bv(out[1]);
    out[0] = nested_bv(container, 0u, output_length(measurement, 0u));
    out[1] = nested_bv(container, 16u, output_length(measurement, 1u));
    memcpy(snapshot, container->bytes, sizeof(snapshot));
    et_c2_checkpoint_load_test_reset_v1();
    CHECK(stage(path, measurement, out, result) == 1);
    CHECK(et_c2_checkpoint_load_test_copy_count_v1() == 0u);
    CHECK(memcmp(snapshot, container->bytes, sizeof(snapshot)) == 0);
    CHECK(bv_canaries(container));
    for (i = 2u; i < 8u; ++i) free_bv(out[i]);
    free_bv(container);
  }

  allocate_outputs(measurement, out, 0x29u);
  {
    const size_t path_length = strlen(argv[1]);
    bv *container = make_bv(192u, 0x29u);
    bv *overlap_path;
    uint8_t snapshot[192];
    free_bv(out[0]);
    out[0] = nested_bv(container, 0u, output_length(measurement, 0u));
    overlap_path = nested_bv(container, 32u, path_length);
    memcpy(overlap_path->bytes, argv[1], path_length);
    memcpy(snapshot, container->bytes, sizeof(snapshot));
    et_c2_checkpoint_load_test_reset_v1();
    CHECK(stage(overlap_path, measurement, out, result) == 1);
    CHECK(et_c2_checkpoint_load_test_copy_count_v1() == 0u);
    CHECK(memcmp(snapshot, container->bytes, sizeof(snapshot)) == 0);
    CHECK(bv_canaries(container));
    for (i = 1u; i < 8u; ++i) free_bv(out[i]);
    free_bv(container);
  }

  allocate_outputs(measurement, out, 0x35u);
  {
    uint8_t snapshot[ET_C2_CHECKPOINT_LOAD_RESULT_BYTES];
    free_bv(out[7]);
    out[7] = nested_bv(measurement, 160u, output_length(measurement, 7u));
    memcpy(snapshot, measurement->bytes, sizeof(snapshot));
    et_c2_checkpoint_load_test_reset_v1();
    CHECK(stage(path, measurement, out, result) == 1);
    CHECK(et_c2_checkpoint_load_test_copy_count_v1() == 0u);
    CHECK(memcmp(snapshot, measurement->bytes, sizeof(snapshot)) == 0);
    for (i = 0u; i < 7u; ++i) free_bv(out[i]);
  }
  CHECK(measure(path, measurement) == 0);

  allocate_outputs(measurement, out, 0x46u);
  {
    uint8_t snapshot[ET_C2_CHECKPOINT_LOAD_RESULT_BYTES];
    memset(result->bytes, 0x91, (size_t)result->length);
    free_bv(out[7]);
    out[7] = nested_bv(result, 160u, output_length(measurement, 7u));
    memcpy(snapshot, result->bytes, sizeof(snapshot));
    et_c2_checkpoint_load_test_reset_v1();
    CHECK(stage(path, measurement, out, result) == 1);
    CHECK(et_c2_checkpoint_load_test_copy_count_v1() == 0u);
    CHECK(memcmp(snapshot, result->bytes, sizeof(snapshot)) == 0);
    for (i = 0u; i < 7u; ++i) free_bv(out[i]);
  }

  allocate_outputs(measurement, out, 0x57u);
  {
    bv *bad_measurement = clone_bv(measurement);
    bad_measurement->bytes[0] ^= 1u;
    check_atomic_failure(path, bad_measurement, out, result, 0x57u);
    free_bv(bad_measurement);
  }
  free_outputs(out);

  /* Complete measurement control-span matrix: NULL, short, long, bad magic. */
  allocate_outputs(measurement, out, 0x58u);
  {
    uint8_t result_snapshot[ET_C2_CHECKPOINT_LOAD_RESULT_BYTES];
    memcpy(result_snapshot, result->bytes, sizeof(result_snapshot));
    et_c2_checkpoint_load_test_reset_v1();
    CHECK(stage(path, NULL, out, result) == 1);
    CHECK(et_c2_checkpoint_load_test_copy_count_v1() == 0u);
    CHECK(outputs_unchanged(out, 0x58u));
    CHECK(memcmp(result_snapshot, result->bytes, sizeof(result_snapshot)) == 0);
  }
  free_outputs(out);
  for (i = 0u; i < 2u; ++i) {
    bv *bad_measurement = clone_bv(measurement);
    allocate_outputs(measurement, out, 0x59u);
    bad_measurement->length += i == 0u ? -1 : 1;
    check_atomic_failure(path, bad_measurement, out, result, 0x59u);
    free_outputs(out);
    free_bv(bad_measurement);
  }

  /* Complete result control-span matrix: NULL, short, and long. */
  for (i = 0u; i < 3u; ++i) {
    const int64_t saved_length = result->length;
    allocate_outputs(measurement, out, 0x68u);
    if (i != 0u) result->length = saved_length + (i == 1u ? -1 : 1);
    et_c2_checkpoint_load_test_reset_v1();
    CHECK(stage(path, measurement, out, i == 0u ? NULL : result) == 1);
    CHECK(et_c2_checkpoint_load_test_copy_count_v1() == 0u);
    CHECK(outputs_unchanged(out, 0x68u));
    result->length = saved_length;
    free_outputs(out);
  }

  /* Core allocation and every reader I/O stage fail before component copy
   * and leave retained-image/reader/fd live counts at zero. */
  allocate_outputs(measurement, out, 0x6au);
  et_c2_checkpoint_core_test_reset_v1();
  et_c2_checkpoint_reader_test_reset_v1();
  et_c2_checkpoint_core_test_fail_alloc_after_v1(0u);
  check_io_failure(path, measurement, out, result, 0x6au, 10);
  et_c2_checkpoint_core_test_reset_v1();
  free_outputs(out);
  {
    struct reader_fault {
      uint32_t stage;
      uint32_t occurrence;
    };
    static const struct reader_fault faults[] = {
        {ET_CHECKPOINT_IO_STAGE_OPEN_SOURCE, 1u},
        {ET_CHECKPOINT_IO_STAGE_STAT_SOURCE, 1u},
        {ET_CHECKPOINT_IO_STAGE_ALLOCATE, 1u},
        {ET_CHECKPOINT_IO_STAGE_READ_SOURCE, 1u},
        {ET_CHECKPOINT_IO_STAGE_STAT_SOURCE, 2u},
        {ET_CHECKPOINT_IO_STAGE_CLOSE_SOURCE, 1u}};
    size_t fault;
    for (fault = 0u; fault < sizeof(faults) / sizeof(faults[0]); ++fault) {
      allocate_outputs(measurement, out, 0x6bu);
      et_c2_checkpoint_reader_test_reset_v1();
      et_c2_checkpoint_reader_test_fail_v1(
          faults[fault].stage, faults[fault].occurrence, EIO);
      check_io_failure(path, measurement, out, result, 0x6bu, 10);
      et_c2_checkpoint_reader_test_reset_v1();
      free_outputs(out);
    }
  }
  allocate_outputs(measurement, out, 0x6cu);
  et_c2_checkpoint_reader_test_reset_v1();
  et_c2_checkpoint_reader_test_set_short_read_v1(0u);
  check_io_failure(path, measurement, out, result, 0x6cu, 10);
  et_c2_checkpoint_reader_test_reset_v1();
  free_outputs(out);

  /* A same-size valid replacement is not detected as an immutable snapshot:
   * the second read is authoritative for bytes and scalar projection. */
  CHECK(strlen(argv[4]) + 24u < sizeof(mutable_path));
  snprintf(mutable_path, sizeof(mutable_path), "%s.same-size", argv[4]);
  replace_file(argv[4], mutable_path);
  {
    bv *mutable_bv = path_bv(mutable_path);
    uint8_t *replacement;
    size_t replacement_length;
    CHECK(measure(mutable_bv, measurement) == 0);
    allocate_outputs(measurement, out, 0x6du);
    replace_file(argv[5], mutable_path);
    et_c2_checkpoint_load_test_reset_v1();
    CHECK(stage(mutable_bv, measurement, out, result) == 0);
    CHECK(et_c2_checkpoint_load_test_copy_count_v1() == 8u);
    replacement = read_file(argv[5], &replacement_length);
    check_exact_outputs(replacement, replacement_length, out);
    CHECK(u64(result->bytes + 104u) == UINT64_MAX);
    CHECK(u64(result->bytes + 112u) == UINT64_C(0x8000000000000000));
    free(replacement);
    free_outputs(out);
    free_bv(mutable_bv);
  }
  CHECK(unlink(mutable_path) == 0);

  et_c2_checkpoint_load_test_reset_v1();
  CHECK(measure(hostile_path, result) == 7);
  CHECK(et_c2_checkpoint_load_test_copy_count_v1() == 0u);
  CHECK(u64(result->bytes + 168u) == 7u);
  CHECK(measure(hostile_i64_path, result) == 7);
  CHECK(et_c2_checkpoint_load_test_copy_count_v1() == 0u);
  CHECK(u64(result->bytes + 168u) == 7u);

  CHECK(strlen(argv[4]) + 16u < sizeof(mutable_path));
  snprintf(mutable_path, sizeof(mutable_path), "%s.mutable", argv[4]);
  replace_file(argv[1], mutable_path);
  {
    bv *mutable_bv = path_bv(mutable_path);
    memset(measurement->bytes, 0, (size_t)measurement->length);
    CHECK(measure(mutable_bv, measurement) == 0);
    allocate_outputs(measurement, out, 0x51u);
    replace_file(argv[4], mutable_path);
    et_c2_checkpoint_load_test_reset_v1();
    status = stage(mutable_bv, measurement, out, result);
    CHECK(status == 2);
    CHECK(u64(result->bytes + 144u) == 5u);
    CHECK(et_c2_checkpoint_load_test_copy_count_v1() == 0u);
    CHECK(outputs_unchanged(out, 0x51u));
    free_outputs(out);
    free_bv(mutable_bv);
  }
  CHECK(unlink(mutable_path) == 0);

  CHECK(bv_canaries(path) && bv_canaries(hostile_path) &&
        bv_canaries(hostile_i64_path) && bv_canaries(measurement) &&
        bv_canaries(result));
  free(valid_file);
  free_bv(path);
  free_bv(hostile_path);
  free_bv(hostile_i64_path);
  free_bv(measurement);
  free_bv(result);
  printf("C2 checkpoint load bridge PASS: %u checks\n", checks);
  return 0;
}
