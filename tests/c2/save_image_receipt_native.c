#define _GNU_SOURCE
#include "c2_save_image_receipt.h"
#include "checkpoint_io.h"

#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static uint8_t *bv(size_t length) {
  uint8_t *value = calloc(1u, length + sizeof(int64_t));
  int64_t declared = (int64_t)length;
  assert(value != NULL);
  memcpy(value, &declared, sizeof(declared));
  return value;
}

static int64_t adopt(uint8_t *image, size_t length, uint8_t *cell) {
  int64_t token = 0;
  assert(et_c2_private_save_image_adopt_v1(image, (int64_t)length, cell) == 0);
  memcpy(&token, cell + sizeof(int64_t), sizeof(token));
  assert(token > 0);
  return token;
}

static uint8_t *busy_digest;
static void check_busy(int64_t token) {
  assert(et_c2_private_save_image_digest_v1(token, busy_digest) == -2);
  assert(et_c2_private_save_image_release_v1(token) == -2);
  assert(et_c2_private_save_image_publish_v1(token, "/tmp/never", 0) == -2);
}

static void check_failure_cuts(uint8_t *image, uint8_t *cell,
                               int64_t previous_token) {
  const int64_t adopt_cuts[] = {
      ET_C2_IMAGE_TEST_FAIL_ENTRY_ALLOC,
      ET_C2_IMAGE_TEST_FAIL_VECTOR_ALLOC,
      ET_C2_IMAGE_TEST_FAIL_AFTER_SHA};
  for (unsigned int repeat = 0; repeat < 32u; ++repeat) {
    for (size_t cut = 0; cut < sizeof(adopt_cuts) / sizeof(adopt_cuts[0]);
         ++cut) {
      memset(cell + sizeof(int64_t), 0xa5, 8u);
      et_c2_private_save_image_test_fail_next_v1(adopt_cuts[cut]);
      assert(et_c2_private_save_image_adopt_v1(image, 3, cell) == -3);
      uint64_t output = UINT64_MAX;
      memcpy(&output, cell + sizeof(int64_t), sizeof(output));
      assert(output == 0u);
      assert(et_c2_private_save_image_live_bytes_v1() == 0u);
      assert(et_c2_private_save_image_live_count_v1() == 0u);
    }
    int64_t token = adopt(image, 3u, cell);
    assert(token == previous_token + 1);
    previous_token = token;
    assert(et_c2_private_save_image_live_bytes_v1() == 3u);
    assert(et_c2_private_save_image_live_count_v1() == 1u);
    et_c2_private_save_image_test_fail_next_v1(ET_C2_IMAGE_TEST_FAIL_RELEASE);
    assert(et_c2_private_save_image_release_v1(-1) == -1);
    assert(et_c2_private_save_image_release_v1(token - 1) == -2);
    assert(et_c2_private_save_image_release_v1(token) == -3);
    assert(et_c2_private_save_image_live_bytes_v1() == 3u);
    assert(et_c2_private_save_image_live_count_v1() == 1u);
    assert(et_c2_private_save_image_release_v1(token) == 0);
    assert(et_c2_private_save_image_live_bytes_v1() == 0u);
    assert(et_c2_private_save_image_live_count_v1() == 0u);
  }
}

int main(int argc, char **argv) {
  static const uint8_t abc_sha[32] = {
      0xba,0x78,0x16,0xbf,0x8f,0x01,0xcf,0xea,0x41,0x41,0x40,0xde,
      0x5d,0xae,0x22,0x23,0xb0,0x03,0x61,0xa3,0x96,0x17,0x7a,0x9c,
      0xb4,0x10,0xff,0x61,0xf2,0x00,0x15,0xad};
  const char *tmpdir = getenv("TMPDIR");
  char *directory = NULL, *path = NULL;
  uint8_t *image, *image8, *cell, *digest, *large, *too_large;
  int64_t token, collision, committed, precommit;
  FILE *file;
  uint8_t observed[3];

  assert(argc == 1 ||
         (argc == 2 && strcmp(argv[1], "--print-temp-template") == 0));
  if (tmpdir == NULL || tmpdir[0] == '\0') tmpdir = "/tmp";
  assert(asprintf(&directory, "%s/eshkol-c2-image-receipt-XXXXXX", tmpdir) >= 0);
  if (argc == 2) {
    puts(directory);
    free(directory);
    return 0;
  }
  assert(mkdtemp(directory) != NULL);
  assert(asprintf(&path, "%s/image", directory) >= 0);
  image = bv(3u); image8 = bv(8u);
  cell = bv(8u); digest = bv(32u);
  large = bv(16777216u); too_large = bv(16777217u);
  memcpy(image + 8u, "abc", 3u);
  assert(et_c2_private_save_image_adopt_v1(image, 4, cell) == -1);
  assert(et_c2_private_save_image_adopt_v1(image8, 8, image8) == -1);
  assert(et_c2_private_save_image_adopt_v1(too_large, 16777217, cell) == -1);
  assert(et_c2_private_save_image_adopt_v1(large, 16777216, cell) == 0);
  memcpy(&token, cell + 8u, sizeof(token));
  assert(et_c2_private_save_image_live_bytes_v1() == 16777216u);
  assert(et_c2_private_save_image_release_v1(token) == 0);
  assert(et_c2_private_save_image_live_bytes_v1() == 0u);
  assert(et_c2_private_save_image_live_count_v1() == 0u);
  check_failure_cuts(image, cell, token);
  assert(et_c2_private_save_image_release_v1(token) == -2);
  assert(et_c2_private_save_image_digest_v1(token + 100, digest) == -1);
  assert(et_c2_private_save_image_digest_v1(token, digest) == -2);
  assert(et_c2_private_save_image_digest_v1(token, image) == -1);

  token = adopt(image, 3u, cell);
  et_c2_private_save_image_test_corrupt_v1(token, 0u);
  assert(et_c2_private_save_image_digest_v1(token, digest) == -3);
  assert(et_c2_private_save_image_publish_v1(token, path, 0) == -3);
  assert(et_c2_private_save_image_release_v1(token) == 0);
  assert(access(path, F_OK) != 0);

  token = adopt(image, 3u, cell);
  memcpy(image + 8u, "xyz", 3u);
  assert(et_c2_private_save_image_digest_v1(token, digest) == 0);
  assert(memcmp(digest + 8u, abc_sha, 32u) == 0);
  assert(et_c2_private_save_image_publish_v1(token, path, 1) == -1);
  busy_digest = digest;
  et_c2_private_save_image_test_writer_hook_v1(check_busy);
  assert(et_c2_private_save_image_publish_v1(token, path, 0) == 0);
  et_c2_private_save_image_test_writer_hook_v1(NULL);
  assert(et_c2_private_save_image_live_bytes_v1() == 0u);
  file = fopen(path, "rb");
  assert(file != NULL && fread(observed, 1u, 3u, file) == 3u);
  assert(fclose(file) == 0 && memcmp(observed, "abc", 3u) == 0);
  assert(et_c2_private_save_image_digest_v1(token, digest) == 0);
  assert(memcmp(digest + 8u, abc_sha, 32u) == 0);
  et_c2_private_save_image_test_fail_next_v1(ET_C2_IMAGE_TEST_FAIL_RELEASE);
  assert(et_c2_private_save_image_release_v1(token) == -3);
  assert(et_c2_private_save_image_live_count_v1() == 1u);
  assert(et_c2_private_save_image_release_v1(token) == 0);
  assert(et_c2_private_save_image_release_v1(token) == -2);

  memcpy(image + 8u, "abc", 3u);
  collision = adopt(image, 3u, cell);
  committed = et_c2_private_save_image_publish_v1(collision, path, 0);
  assert(committed > 0 && !et_checkpoint_io_status_committed_v1(committed));
  assert(et_c2_private_save_image_release_v1(collision) == -2);
  assert(et_c2_private_save_image_live_bytes_v1() == 0u);

  assert(unlink(path) == 0);
  precommit = adopt(image, 3u, cell);
  et_checkpoint_io_test_reset_v1();
  et_checkpoint_io_test_fail_v1(ET_CHECKPOINT_IO_STAGE_ALLOCATE, 1, ENOMEM);
  collision = et_c2_private_save_image_publish_v1(precommit, path, 0);
  assert(collision > 0 && !et_checkpoint_io_status_committed_v1(collision));
  assert(et_checkpoint_io_status_stage_v1(collision) ==
         ET_CHECKPOINT_IO_STAGE_ALLOCATE);
  assert(access(path, F_OK) != 0);
  assert(et_c2_private_save_image_release_v1(precommit) == -2);
  assert(et_c2_private_save_image_live_bytes_v1() == 0u);

  committed = adopt(image, 3u, cell);
  et_checkpoint_io_test_reset_v1();
  et_checkpoint_io_test_fail_v1(ET_CHECKPOINT_IO_STAGE_SYNC_DIRECTORY, 1, EIO);
  collision = et_c2_private_save_image_publish_v1(committed, path, 0);
  assert(collision > 0 && et_checkpoint_io_status_committed_v1(collision));
  assert(et_checkpoint_io_status_stage_v1(collision) ==
         ET_CHECKPOINT_IO_STAGE_SYNC_DIRECTORY);
  assert(et_c2_private_save_image_digest_v1(committed, digest) == -2);
  assert(et_c2_private_save_image_live_bytes_v1() == 0u);
  assert(unlink(path) == 0 && rmdir(directory) == 0);
  et_c2_private_save_image_test_set_issued_v1((uint64_t)INT64_MAX);
  assert(et_c2_private_save_image_adopt_v1(image, 3, cell) == -4);
  assert(et_c2_private_save_image_live_bytes_v1() == 0u);
  free(image); free(image8); free(cell); free(digest);
  free(large); free(too_large); free(path); free(directory);
  puts("C2 private image receipt native checks passed");
  return 0;
}
