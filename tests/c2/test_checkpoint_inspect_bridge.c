#include "c2_checkpoint_core.h"
#include "c2_checkpoint_inspect_bridge.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct bytevector_320 {
  int64_t length;
  uint8_t bytes[320];
} bytevector_320;

typedef struct bytevector_path {
  int64_t length;
  uint8_t bytes[4097];
} bytevector_path;

static unsigned checks;
#define CHECK(x) do { ++checks; if (!(x)) { \
  fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); } } while (0)

static uint64_t u64(const uint8_t *p) {
  uint64_t value = 0u;
  unsigned i;
  for (i = 0u; i < 8u; ++i) value |= (uint64_t)p[i] << (8u * i);
  return value;
}

static bytevector_path path_value(const char *path) {
  bytevector_path result;
  size_t length = strlen(path);
  CHECK(length <= 4096u);
  memset(&result, 0, sizeof(result));
  result.length = (int64_t)length;
  memcpy(result.bytes, path, length);
  return result;
}

static int64_t inspect(bytevector_path *path, int64_t file, int64_t metadata,
                       int64_t tensor, int64_t tensors, int64_t profile,
                       bytevector_320 *result) {
  result->length = sizeof(result->bytes);
  memset(result->bytes, 0xa5, sizeof(result->bytes));
  return et_c2_private_checkpoint_inspect_bridge_v1(
      path, file, metadata, tensor, tensors, profile, result);
}

static void expect_format(const char *path_text, int64_t category,
                          uint64_t format_code) {
  bytevector_path path = path_value(path_text);
  bytevector_320 result;
  CHECK(inspect(&path, INT64_C(1099511627776), INT64_C(268435456),
                INT64_C(274877906944), 6826, 0, &result) == category);
  CHECK(u64(result.bytes + 8u) == (uint64_t)category);
  CHECK(u64(result.bytes + 16u) == ET_C2_CORE_CODE_FORMAT);
  CHECK(u64(result.bytes + 40u) == (uint64_t)category);
  CHECK(u64(result.bytes + 48u) == format_code);
  CHECK(et_c2_checkpoint_core_test_live_images_v1() == 0u);
}

int main(int argc, char **argv) {
  bytevector_path valid, missing, embedded, too_long;
  bytevector_320 result;
  CHECK(argc == 7);
  valid = path_value(argv[1]);
  CHECK(inspect(&valid, 3039, 2567, 8, 3, 0, &result) == 0);
  CHECK(u64(result.bytes) == UINT64_C(0x4332494e53503130));
  CHECK(u64(result.bytes + 64u) == 3039u);
  CHECK(u64(result.bytes + 72u) == 24u);
  CHECK(u64(result.bytes + 80u) == 3u);
  CHECK(u64(result.bytes + 88u) == 96u);
  CHECK(memcmp(result.bytes + 112u, "sha256:eshkol-byte-tokenizer-v1:", 32u) == 0);
  CHECK(memcmp(result.bytes + 208u, "sha256:eshkol-config-json-v1:", 29u) == 0);
  CHECK(memcmp(result.bytes + 301u, "0.1.0-draft", 11u) == 0);
  CHECK(et_c2_checkpoint_core_test_live_images_v1() == 0u);

  expect_format(argv[2], ET_C2_CORE_UNSUPPORTED, ET_C2_FORMAT_CODE_IDENTITY);
  expect_format(argv[3], ET_C2_CORE_CORRUPT_DATA, ET_C2_FORMAT_CODE_CHECKSUM);
  expect_format(argv[4], ET_C2_CORE_CORRUPT_DATA, ET_C2_FORMAT_CODE_IDENTITY);

  CHECK(inspect(&valid, 3038, 2567, 8, 3, 0, &result) ==
        ET_C2_CORE_CORRUPT_DATA);
  CHECK(u64(result.bytes + 16u) == ET_C2_CORE_CODE_PHYSICAL_SIZE);
  CHECK(et_c2_checkpoint_core_test_live_images_v1() == 0u);

  { bytevector_path profile = path_value(argv[5]);
    CHECK(inspect(&profile, INT64_C(1099511627776), INT64_C(268435456),
                  INT64_C(274877906944), 6826, 1, &result) ==
          ET_C2_CORE_UNSUPPORTED);
    CHECK(u64(result.bytes + 16u) == ET_C2_CORE_CODE_FORMAT);
    CHECK(u64(result.bytes + 48u) == ET_C2_FORMAT_CODE_PROFILE);
    CHECK(u64(result.bytes + 56u) == 112u);
  }

  missing = path_value(argv[6]);
  CHECK(inspect(&missing, 3039, 2567, 8, 3, 0, &result) == ET_C2_CORE_IO);
  CHECK(u64(result.bytes + 16u) == ET_C2_CORE_CODE_NATIVE_IO);
  CHECK(u64(result.bytes + 32u) != 0u);

  embedded = valid;
  embedded.bytes[1] = 0;
  CHECK(inspect(&embedded, 3039, 2567, 8, 3, 0, &result) ==
        ET_C2_CORE_INVALID_ARGUMENT);
  memset(&too_long, 'x', sizeof(too_long));
  too_long.length = 4097;
  CHECK(inspect(&too_long, 3039, 2567, 8, 3, 0, &result) ==
        ET_C2_CORE_INVALID_ARGUMENT);
  CHECK(inspect(&valid, 0, 2567, 8, 3, 0, &result) ==
        ET_C2_CORE_INVALID_ARGUMENT);

  et_c2_checkpoint_core_test_fail_alloc_after_v1(0u);
  CHECK(inspect(&valid, 3039, 2567, 8, 3, 0, &result) == ET_C2_CORE_IO);
  CHECK(u64(result.bytes + 16u) == ET_C2_CORE_CODE_STAGING_ALLOCATION);
  CHECK(et_c2_checkpoint_core_test_live_images_v1() == 0u);
  et_c2_checkpoint_core_test_reset_v1();

  et_c2_checkpoint_inspect_test_fail_projection_v1(1);
  CHECK(inspect(&valid, 3039, 2567, 8, 3, 0, &result) == ET_C2_CORE_INTERNAL);
  CHECK(u64(result.bytes + 16u) == ET_C2_INSPECT_CODE_PROJECTION);
  CHECK(et_c2_checkpoint_core_test_live_images_v1() == 0u);
  et_c2_checkpoint_inspect_test_reset_v1();

  et_c2_checkpoint_inspect_test_fail_release_v1(1);
  CHECK(inspect(&valid, 3039, 2567, 8, 3, 0, &result) == ET_C2_CORE_INTERNAL);
  CHECK(u64(result.bytes) == 0u);
  CHECK(u64(result.bytes + 16u) == ET_C2_INSPECT_CODE_RELEASE);
  CHECK(et_c2_checkpoint_core_test_live_images_v1() == 0u);
  et_c2_checkpoint_inspect_test_reset_v1();

  result.length = 319;
  CHECK(et_c2_private_checkpoint_inspect_bridge_v1(
            &valid, 3039, 2567, 8, 3, 0, &result) ==
        ET_C2_CORE_INVALID_ARGUMENT);
  CHECK(et_c2_checkpoint_core_test_live_images_v1() == 0u);
  printf("C2 private checkpoint inspect bridge: PASS (%u checks)\n", checks);
  return 0;
}
