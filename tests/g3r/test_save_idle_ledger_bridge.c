#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "save_idle_ledger_bridge.c"

static int generator, snapshot;
static int create_calls, copy_calls, release_calls;
static int copy_error, release_error;

void *et_g3r_private_g3t_idle_snapshot_create_v1(void *candidate) {
  ++create_calls;
  return candidate == &generator ? &snapshot : NULL;
}

int64_t et_g3r_private_g3t_idle_snapshot_copy_v1(
    void *candidate, et_g3r_g3t_idle_snapshot_words_v1 *destination,
    int64_t bytes) {
  ++copy_calls;
  assert(candidate == &snapshot && bytes == 144);
  if (copy_error) return -1;
  for (int i = 0; i < 18; ++i) destination->word[i] = i + 17;
  return 0;
}

int64_t et_g3r_private_g3t_idle_snapshot_release_v1(void *candidate) {
  ++release_calls;
  assert(candidate == &snapshot);
  return release_error ? -1 : 0;
}

int main(void) {
  unsigned char output[152];
  int64_t length = 144;
  memset(output, 0xa5, sizeof(output));
  memcpy(output, &length, sizeof(length));

  assert(et_g3r_test_save_idle_ledger_copy_v1(
             &generator, output, 143) == -1);
  assert(create_calls == 0);
  length = 143;
  memcpy(output, &length, sizeof(length));
  assert(et_g3r_test_save_idle_ledger_copy_v1(
             &generator, output, 144) == -1);
  assert(create_calls == 0);
  length = 144;
  memcpy(output, &length, sizeof(length));
  assert(et_g3r_test_save_idle_ledger_copy_v1(
             &snapshot, output, 144) == -1);
  assert(create_calls == 1 && copy_calls == 0 && release_calls == 0);

  copy_error = 1;
  assert(et_g3r_test_save_idle_ledger_copy_v1(
             &generator, output, 144) == -1);
  assert(release_calls == 1 && output[8] == 0xa5);
  copy_error = 0;
  release_error = 1;
  assert(et_g3r_test_save_idle_ledger_copy_v1(
             &generator, output, 144) == -1);
  assert(release_calls == 2 && output[8] == 0xa5);
  release_error = 0;

  assert(et_g3r_test_save_idle_ledger_copy_v1(
             &generator, output, 144) == 0);
  assert(create_calls == 4 && copy_calls == 3 && release_calls == 3);
  for (int i = 0; i < 18; ++i) {
    int64_t observed;
    memcpy(&observed, output + 8 + 8 * i, sizeof(observed));
    assert(observed == i + 17);
  }
  return 0;
}
