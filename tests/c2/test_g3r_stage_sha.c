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

typedef struct bv { int64_t length; uint8_t bytes[]; } bv;
static unsigned checks;
#define CHECK(x) do { ++checks; if (!(x)) { \
  fprintf(stderr, "G3R C2 STAGE SHA FAIL %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

static bv *new_bv(size_t n, uint8_t fill) {
  bv *b = malloc(sizeof(*b) + n);
  CHECK(b != NULL);
  b->length = (int64_t)n;
  memset(b->bytes, fill, n);
  return b;
}
static bv *path_bv(const char *path) {
  size_t n = strlen(path);
  bv *b = new_bv(n, 0);
  memcpy(b->bytes, path, n);
  return b;
}
static bv *nested(bv *parent, size_t offset, size_t n) {
  bv *b;
  CHECK(offset + sizeof(bv) + n <= (size_t)parent->length);
  b = (bv *)(void *)(parent->bytes + offset);
  b->length = (int64_t)n;
  return b;
}
static uint64_t u64(const uint8_t *p) {
  uint64_t v = 0;
  for (unsigned i = 0; i < 8; ++i) v |= (uint64_t)p[i] << (i * 8u);
  return v;
}
static size_t out_length(const bv *measure, size_t i) {
  static const size_t at[8] = {16, 0, 24, 32, 40, 48, 56, 64};
  return i == 1 ? 93u : (size_t)u64(measure->bytes + at[i]);
}
static void make_outputs(const bv *measure, bv *out[8], uint8_t fill) {
  for (size_t i = 0; i < 8; ++i) out[i] = new_bv(out_length(measure, i), fill);
}
static void free_outputs(bv *out[8]) {
  for (size_t i = 0; i < 8; ++i) free(out[i]);
}
static int outputs_equal_fill(const bv *out[8], uint8_t fill) {
  for (size_t i = 0; i < 8; ++i)
    for (size_t j = 0; j < (size_t)out[i]->length; ++j)
      if (out[i]->bytes[j] != fill) return 0;
  return 1;
}
static int64_t measure(const bv *path, bv *result) {
  return et_c2_private_checkpoint_load_measure_v1(
      path, 1048576, 1048576, 1048576, 16, 0, result);
}
static int64_t stage(const bv *path, const bv *measure_result,
                     bv *out[8], bv *result) {
  return et_c2_private_checkpoint_load_stage_with_sha_v1(
      path, 1048576, 1048576, 1048576, 16, 0, measure_result,
      out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], result);
}
static int64_t old_stage(const bv *path, const bv *measure_result,
                         bv *out[8], bv *result) {
  return et_c2_private_checkpoint_load_stage_v1(
      path, 1048576, 1048576, 1048576, 16, 0, measure_result,
      out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], result);
}
static int hex_digit(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  return -1;
}
static void check_sha(const bv *result, const char *hex) {
  CHECK(strlen(hex) == 64);
  for (size_t i = 0; i < 32; ++i)
    CHECK(result->bytes[192 + i] ==
          (uint8_t)((hex_digit(hex[2 * i]) << 4) | hex_digit(hex[2 * i + 1])));
}
static void replace_file(const char *source, const char *destination) {
  FILE *in = fopen(source, "rb"), *out = fopen(destination, "wb");
  uint8_t block[4096];
  size_t n;
  CHECK(in && out);
  while ((n = fread(block, 1, sizeof(block), in)) != 0)
    CHECK(fwrite(block, 1, n, out) == n);
  CHECK(!ferror(in));
  CHECK(fclose(in) == 0 && fclose(out) == 0);
}
static void check_no_live_image(void) {
  CHECK(et_c2_checkpoint_core_test_live_images_v1() == 0);
  CHECK(et_c2_checkpoint_reader_test_live_count_v1() == 0);
  CHECK(et_c2_checkpoint_reader_test_fd_live_count_v1() == 0);
}
static void check_failure_record(const bv *result, uint64_t category,
                                 uint64_t code) {
  CHECK(u64(result->bytes + 136) == category);
  CHECK(u64(result->bytes + 144) == code);
  for (size_t i = 192; i < 224; ++i) CHECK(result->bytes[i] == 0);
}

int main(int argc, char **argv) {
  bv *path, *measurement, *result, *old_result, *out[8], *old_out[8];
  CHECK(argc == 7);
  path = path_bv(argv[1]);
  measurement = new_bv(192, 0xa5);
  result = new_bv(224, 0xa5);
  old_result = new_bv(192, 0xa5);
  CHECK(measure(path, measurement) == 0);
  make_outputs(measurement, out, 0x46);
  make_outputs(measurement, old_out, 0x46);
  CHECK(stage(path, measurement, out, result) == 0);
  CHECK(old_stage(path, measurement, old_out, old_result) == 0);
  CHECK(memcmp(result->bytes, old_result->bytes, 192) == 0);
  for (size_t i = 0; i < 8; ++i) {
    CHECK(out[i]->length == old_out[i]->length);
    CHECK(memcmp(out[i]->bytes, old_out[i]->bytes,
                 (size_t)out[i]->length) == 0);
  }
  check_sha(result, argv[4]);
  free_outputs(out);
  free_outputs(old_out);

  /* Invalid header/alias admission preserves even an already-filled result. */
  make_outputs(measurement, out, 0x51);
  for (int delta = -1; delta <= 1; delta += 2) {
    uint8_t before[224];
    memcpy(before, result->bytes, 224);
    result->length = 224 + delta;
    CHECK(stage(path, measurement, out, result) == 1);
    result->length = 224;
    CHECK(memcmp(result->bytes, before, 224) == 0);
    CHECK(outputs_equal_fill((const bv **)out, 0x51));
  }
  {
    uint8_t path_before[4096], measurement_before[192];
    CHECK((size_t)path->length <= sizeof(path_before));
    memcpy(path_before, path->bytes, (size_t)path->length);
    memcpy(measurement_before, measurement->bytes, sizeof(measurement_before));
    CHECK(stage(path, measurement, out, path) == 1);
    CHECK(stage(path, measurement, out, measurement) == 1);
    for (size_t i = 0; i < 8; ++i)
      CHECK(stage(path, measurement, out, out[i]) == 1);
    CHECK(memcmp(path_before, path->bytes, (size_t)path->length) == 0);
    CHECK(memcmp(measurement_before, measurement->bytes,
                 sizeof(measurement_before)) == 0);
    CHECK(outputs_equal_fill((const bv **)out, 0x51));
  }
  for (size_t i = 0; i < 8; ++i) {
    bv *container = new_bv(out_length(measurement, i) + 256, 0x6d);
    bv *saved = out[i], *aliased_result = nested(container, 0, 224);
    bv *inside = nested(container, 40, out_length(measurement, i));
    uint8_t *before = malloc((size_t)container->length);
    CHECK(before != NULL);
    memcpy(before, container->bytes, (size_t)container->length);
    out[i] = inside;
    CHECK(stage(path, measurement, out, aliased_result) == 1);
    CHECK(memcmp(before, container->bytes, (size_t)container->length) == 0);
    out[i] = saved;
    free(before);
    free(container);
  }
  for (int alias = 0; alias < 2; ++alias) {
    bv *container = new_bv(512, 0x6d);
    bv *aliased_result = nested(container, 0, 224);
    bv *inside = nested(container, 40,
                        alias == 0 ? (size_t)path->length : 192u);
    uint8_t before[512];
    if (alias == 0) memcpy(inside->bytes, path->bytes, (size_t)path->length);
    else memcpy(inside->bytes, measurement->bytes, 192);
    memcpy(before, container->bytes, sizeof(before));
    CHECK(stage(alias == 0 ? inside : path,
                alias == 0 ? measurement : inside, out, aliased_result) == 1);
    CHECK(memcmp(before, container->bytes, sizeof(before)) == 0);
    CHECK(outputs_equal_fill((const bv **)out, 0x51));
    free(container);
  }
  CHECK(outputs_equal_fill((const bv **)out, 0x51));
  check_no_live_image();
  free_outputs(out);

  /* A post-validation failure preserves C2's first 192 diagnostic bytes and
   * scrubs the private digest tail; no component or image owner survives. */
  make_outputs(measurement, out, 0x72);
  memset(result->bytes, 0xa5, 224);
  et_c2_checkpoint_load_test_reset_v1();
  et_c2_checkpoint_load_test_fail_after_validate_v1(1);
  CHECK(stage(path, measurement, out, result) == 11);
  CHECK(old_stage(path, measurement, out, old_result) == 11);
  CHECK(memcmp(result->bytes, old_result->bytes, 192) == 0);
  check_failure_record(result, 11, ET_C2_CORE_CODE_INVALID_IMAGE);
  CHECK(outputs_equal_fill((const bv **)out, 0x72));
  CHECK(et_c2_checkpoint_load_test_copy_count_v1() == 0);
  check_no_live_image();
  et_c2_checkpoint_load_test_reset_v1();
  free_outputs(out);

  make_outputs(measurement, out, 0x73);
  memset(result->bytes, 0xa5, 224);
  et_c2_checkpoint_load_test_fail_after_release_v1(1);
  CHECK(stage(path, measurement, out, result) == 11);
  CHECK(old_stage(path, measurement, out, old_result) == 11);
  CHECK(memcmp(result->bytes, old_result->bytes, 192) == 0);
  check_failure_record(result, 11, ET_C2_CORE_CODE_INVALID_IMAGE);
  CHECK(out[5]->bytes[0] != 0x73);
  check_no_live_image();
  et_c2_checkpoint_load_test_reset_v1();
  free_outputs(out);

  make_outputs(measurement, out, 0x74);
  memset(result->bytes, 0xa5, 224);
  et_c2_checkpoint_core_test_fail_alloc_after_v1(0);
  CHECK(stage(path, measurement, out, result) == 10);
  CHECK(old_stage(path, measurement, out, old_result) == 10);
  CHECK(memcmp(result->bytes, old_result->bytes, 192) == 0);
  CHECK(outputs_equal_fill((const bv **)out, 0x74));
  CHECK(et_c2_checkpoint_load_test_copy_count_v1() == 0);
  check_failure_record(result, 10, ET_C2_CORE_CODE_STAGING_ALLOCATION);
  check_no_live_image();
  et_c2_checkpoint_core_test_reset_v1();
  free_outputs(out);

  make_outputs(measurement, out, 0x76);
  memset(result->bytes, 0xa5, 224);
  et_c2_checkpoint_reader_test_reset_v1();
  et_c2_checkpoint_reader_test_fail_v1(ET_CHECKPOINT_IO_STAGE_READ_SOURCE,
                                        1, EIO);
  CHECK(stage(path, measurement, out, result) == 10);
  CHECK(outputs_equal_fill((const bv **)out, 0x76));
  check_failure_record(result, 10, ET_C2_CORE_CODE_NATIVE_IO);
  check_no_live_image();
  et_c2_checkpoint_reader_test_reset_v1();
  free_outputs(out);

  /* Measure valid B, replace same-size bytes with valid C, then stage: both
   * the content projection and digest must come from C, never B. */
  replace_file(argv[2], argv[6]);
  {
    bv *mutable_path = path_bv(argv[6]);
    bv *b_measure = new_bv(192, 0);
    CHECK(measure(mutable_path, b_measure) == 0);
    make_outputs(b_measure, out, 0x5a);
    replace_file(argv[3], argv[6]);
    CHECK(stage(mutable_path, b_measure, out, result) == 0);
    check_sha(result, argv[5]);
    CHECK(u64(result->bytes + 104) == UINT64_MAX);
    CHECK(u64(result->bytes + 112) == UINT64_C(0x8000000000000000));
    CHECK(out[5]->bytes[0] != 0x5a);
    {
      bv *direct_result = new_bv(192, 0);
      bv *direct_out[8];
      uint8_t staged_result[224];
      make_outputs(b_measure, direct_out, 0);
      CHECK(old_stage(mutable_path, b_measure, direct_out, direct_result) == 0);
      CHECK(memcmp(result->bytes, direct_result->bytes, 192) == 0);
      for (size_t i = 0; i < 8; ++i)
        CHECK(memcmp(out[i]->bytes, direct_out[i]->bytes,
                     (size_t)out[i]->length) == 0);
      memcpy(staged_result, result->bytes, sizeof(staged_result));
      replace_file(argv[2], argv[6]);
      CHECK(memcmp(result->bytes, staged_result, sizeof(staged_result)) == 0);
      for (size_t i = 0; i < 8; ++i)
        CHECK(memcmp(out[i]->bytes, direct_out[i]->bytes,
                     (size_t)out[i]->length) == 0);
      free_outputs(direct_out);
      free(direct_result);
    }
    free_outputs(out);
    free(b_measure);
    free(mutable_path);
  }
  replace_file(argv[1], argv[6]);
  {
    bv *mutable_path = path_bv(argv[6]);
    make_outputs(measurement, out, 0x75);
    replace_file(argv[2], argv[6]);
    memset(result->bytes, 0xa5, 224);
    CHECK(stage(mutable_path, measurement, out, result) == 2);
    check_failure_record(result, 2, ET_C2_CORE_CODE_PROBED_HEADER_CHANGED);
    CHECK(outputs_equal_fill((const bv **)out, 0x75));
    check_no_live_image();
    free_outputs(out);
    free(mutable_path);
  }
  replace_file(argv[1], argv[6]);
  {
    bv *mutable_path = path_bv(argv[6]);
    FILE *file = fopen(argv[6], "rb+");
    CHECK(file != NULL);
    CHECK(ftruncate(fileno(file), 80) == 0);
    CHECK(fclose(file) == 0);
    make_outputs(measurement, out, 0x77);
    memset(result->bytes, 0xa5, 224);
    CHECK(stage(mutable_path, measurement, out, result) != 0);
    CHECK(outputs_equal_fill((const bv **)out, 0x77));
    for (size_t i = 192; i < 224; ++i) CHECK(result->bytes[i] == 0);
    check_no_live_image();
    free_outputs(out);
    free(mutable_path);
  }
  CHECK(unlink(argv[6]) == 0);
  check_no_live_image();
  free(path); free(measurement); free(result); free(old_result);
  printf("G3R C2 stage SHA PASS: %u checks\n", checks);
  return 0;
}
