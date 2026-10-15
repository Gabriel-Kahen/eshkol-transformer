#include "eshkol_transformer/g3t_transport.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>

int64_t et_g3r_private_held_image_copy_v1(void *hold, void *header);

static int64_t calls;
static void *seen_hold;
static int64_t seen_bytes;
static int64_t native_status;

int64_t et_g3r_private_g3t_save_copy_v1(
    void *hold, et_g3r_g3t_held_words_v1 *destination,
    int64_t destination_bytes) {
  ++calls;
  seen_hold = hold;
  seen_bytes = destination_bytes;
  if (native_status) return native_status;
  for (int i = 0; i < 17; ++i) destination->word[i] = i + 100;
  return 0;
}

int main(void) {
  struct {
    int64_t length;
    et_g3r_g3t_held_words_v1 image;
    unsigned char sentinel[8];
  } bytes;
  memset(&bytes, 0xa5, sizeof(bytes));
  unsigned char before[sizeof(bytes)];
  memcpy(before, &bytes, sizeof(bytes));
  void *hold = &bytes;
  assert(et_g3r_private_held_image_copy_v1(hold, NULL) != 0);
  assert(calls == 0);
  bytes.length = 135;
  assert(et_g3r_private_held_image_copy_v1(hold, &bytes) != 0);
  bytes.length = 137;
  assert(et_g3r_private_held_image_copy_v1(hold, &bytes) != 0);
  assert(calls == 0);
  bytes.length = 136;
  native_status = -2;
  assert(et_g3r_private_held_image_copy_v1(hold, &bytes) == -2);
  assert(calls == 1 && seen_hold == hold && seen_bytes == 136);
  assert(!memcmp(&bytes.image, before + sizeof(int64_t),
                 sizeof(bytes.image)));
  native_status = 0;
  assert(et_g3r_private_held_image_copy_v1(hold, &bytes) == 0);
  assert(calls == 2 && seen_hold == hold && seen_bytes == 136);
  for (int i = 0; i < 17; ++i) assert(bytes.image.word[i] == i + 100);
  assert(!memcmp(bytes.sentinel, before + sizeof(int64_t) +
                 sizeof(bytes.image), sizeof(bytes.sentinel)));
  return 0;
}
