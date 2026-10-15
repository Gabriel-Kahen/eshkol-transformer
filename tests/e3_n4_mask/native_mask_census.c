#include "d2_native.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t u32le(const unsigned char *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
         ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static int census(const char *root, const char *name, int count) {
  char path[1024];
  if (snprintf(path, sizeof(path), "%s/%s/shard-0000000000000000.ets",
               root, name) >= (int)sizeof(path)) return 2;
  FILE *file = fopen(path, "rb");
  if (!file) return 2;
  unsigned char shard[256];
  size_t size = fread(shard, 1, sizeof(shard), file);
  if (!feof(file) || fclose(file) != 0 || size < 64) return 2;
  uint32_t header = u32le(shard + 12);
  if (header < 48 || (size_t)header + (size_t)count * 8u + 32u != size)
    return 2;
  struct { int64_t length; unsigned char data[256]; } tagged = {0};
  tagged.length = (int64_t)size;
  memcpy(tagged.data, shard, size);
  int owner = 0;
  if (et_d2_dataset_open_v1(&owner, 0) != 0) return 2;
  int64_t generation = et_d2_batch_create_v1(&owner, 1, 2, 34);
  if (generation <= 0 || et_d2_batch_write_pair_i64le_source_span_v1(
          &owner, generation, 0, &tagged, (int64_t)size, header,
          &tagged, (int64_t)size, header + 8, count - 1) != 0 ||
      et_d2_batch_seal_v1(&owner, generation) != 0) return 2;
  int64_t lease = et_d2_batch_borrow_begin_v1(&owner, generation);
  const et_kernel_tensor_view_v1 *view =
      et_d2_batch_borrow_loss_mask_v1(&owner, generation, lease);
  if (lease <= 0 || !view || view->byte_length != 2u || !view->data) return 2;
  const unsigned char *mask = view->data;
  unsigned char actual[2] = {mask[0], mask[1]};
  if (et_d2_batch_borrow_end_v1(&owner, generation, lease) != 0 ||
      et_d2_batch_release_v1(&owner, generation) != 0 ||
      et_d2_dataset_close_v1(&owner) != 0) return 2;
  unsigned char expected[2] = {1u, (unsigned char)(count == 3)};
  if (memcmp(actual, expected, 2) != 0) {
    printf("MASK FIRST DIVERGENCE %s %u%u expected %u%u\n", name,
           (unsigned)actual[0], (unsigned)actual[1],
           (unsigned)expected[0], (unsigned)expected[1]);
    return 71;
  }
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  const char *names[] = {"p2", "u2", "p3", "u3"};
  for (size_t i = 0; i < 4; ++i) {
    int result = census(argv[1], names[i], i < 2 ? 2 : 3);
    if (result) return result;
  }
  puts("E3 N4 D1/D2 MASK CENSUS PASS: 4 rows");
  return 0;
}
