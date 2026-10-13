/* Feature-private detached C2 SAVE image owner. Not linked by public C2. */
#include "c2_checkpoint_format.h"
#include "checkpoint_io.h"

#include <limits.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef ET_C2_SAVE_IMAGE_RECEIPT_PRIVATE
#include "c2_checkpoint_load_sha.inc"

enum { IMAGE_READY = 1, IMAGE_WRITING = 2, IMAGE_PUBLISHED = 3 };
typedef struct image_owner {
  struct image_owner *next;
  int64_t token;
  uint8_t *vector;
  uint64_t length;
  uint8_t digest[32];
  int state;
} image_owner;

static atomic_flag image_lock = ATOMIC_FLAG_INIT;
static image_owner *image_live;
static uint64_t image_issued;
#ifdef ET_C2_SAVE_IMAGE_RECEIPT_TESTING
static void (*image_writer_hook)(int64_t);
#endif
static void lock_images(void) {
  while (atomic_flag_test_and_set_explicit(&image_lock, memory_order_acquire)) {}
}
static void unlock_images(void) {
  atomic_flag_clear_explicit(&image_lock, memory_order_release);
}
static void wipe(void *ptr, size_t count) {
  volatile uint8_t *bytes = (volatile uint8_t *)ptr;
  while (count-- != 0u) *bytes++ = 0u;
}
static void hash_image(const uint8_t *bytes, size_t length, uint8_t out[32]) {
  c2_stage_sha_state state;
  c2_stage_sha_init(&state);
  c2_stage_sha_update(&state, bytes, length);
  c2_stage_sha_final(&state, out);
  wipe(&state, sizeof(state));
}
static int span(const void *opaque, uint64_t exact, const uint8_t **bytes) {
  int64_t declared;
  uintptr_t start;
  if (opaque == NULL || exact > (uint64_t)INT64_MAX) return 0;
  memcpy(&declared, opaque, sizeof(declared));
  start = (uintptr_t)opaque;
  if (declared < 0 || (uint64_t)declared != exact ||
      start > UINTPTR_MAX - sizeof(declared) ||
      exact > UINTPTR_MAX - start - sizeof(declared)) return 0;
  *bytes = (const uint8_t *)(start + sizeof(declared));
  return 1;
}
static int overlap(const void *a, uint64_t alen,
                   const void *b, uint64_t blen) {
  uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
  uint64_t ax = alen + sizeof(int64_t), by = blen + sizeof(int64_t);
  if (ax > UINTPTR_MAX - x || by > UINTPTR_MAX - y) return 1;
  return x < y + by && y < x + ax;
}
static image_owner *find_image(int64_t token, image_owner **previous) {
  image_owner *entry = image_live, *prior = NULL;
  while (entry != NULL && entry->token != token) {
    prior = entry;
    entry = entry->next;
  }
  if (previous != NULL) *previous = prior;
  return entry;
}
static void unlink_image(image_owner *entry, image_owner *previous) {
  if (previous != NULL) previous->next = entry->next;
  else image_live = entry->next;
}
static void free_image(image_owner *entry) {
  if (entry->vector != NULL) {
    wipe(entry->vector, (size_t)entry->length + sizeof(int64_t));
    free(entry->vector);
    entry->vector = NULL;
  }
}
static int valid_token(int64_t token) {
  return token > 0 && (uint64_t)token <= image_issued;
}

int64_t et_c2_private_save_image_adopt_v1(const void *encoded_bv,
                                           int64_t length, void *token_bv8) {
  const uint8_t *source, *output;
  image_owner *entry;
  uint8_t digest[32];
  if (length <= 0 || (uint64_t)length > ET_C2_PROFILE_MAX_FILE_BYTES ||
      !span(encoded_bv, (uint64_t)length, &source) ||
      !span(token_bv8, 8u, &output) ||
      overlap(encoded_bv, (uint64_t)length, token_bv8, 8u)) return -1;
  memset((void *)(uintptr_t)output, 0, 8u);
  entry = (image_owner *)calloc(1u, sizeof(*entry));
  if (entry == NULL) return -3;
  entry->vector = (uint8_t *)malloc((size_t)length + sizeof(int64_t));
  if (entry->vector == NULL) { free(entry); return -3; }
  memcpy(entry->vector, &length, sizeof(length));
  memcpy(entry->vector + sizeof(length), source, (size_t)length);
  entry->length = (uint64_t)length;
  hash_image(entry->vector + sizeof(length), (size_t)length, digest);
  memcpy(entry->digest, digest, sizeof(digest));
  wipe(digest, sizeof(digest));
  entry->state = IMAGE_READY;
  lock_images();
  if (image_issued == (uint64_t)INT64_MAX) {
    unlock_images(); free_image(entry); wipe(entry, sizeof(*entry)); free(entry);
    return -4;
  }
  entry->token = (int64_t)++image_issued;
  entry->next = image_live;
  image_live = entry;
  memcpy((void *)(uintptr_t)output, &entry->token, sizeof(entry->token));
  unlock_images();
  return 0;
}

int64_t et_c2_private_save_image_digest_v1(int64_t token, void *digest_bv32) {
  const uint8_t *output;
  image_owner *entry;
  uint8_t actual[32];
  if (!span(digest_bv32, 32u, &output) || token <= 0) return -1;
  lock_images();
  if (!valid_token(token)) { unlock_images(); return -1; }
  entry = find_image(token, NULL);
  if (entry == NULL || entry->state == IMAGE_WRITING) {
    unlock_images(); return -2;
  }
  if (entry->state == IMAGE_READY) {
    hash_image(entry->vector + sizeof(int64_t), (size_t)entry->length, actual);
    if (memcmp(actual, entry->digest, 32u) != 0) {
      wipe(actual, sizeof(actual)); unlock_images(); return -3;
    }
    wipe(actual, sizeof(actual));
  }
  memcpy((void *)(uintptr_t)output, entry->digest, 32u);
  unlock_images();
  return 0;
}

int64_t et_c2_private_save_image_publish_v1(int64_t token,
                                              const char *path,
                                              int64_t overwrite) {
  image_owner *entry, *prior;
  int64_t status;
  uint8_t actual[32];
  if (path == NULL || overwrite != 0 || token <= 0) return -1;
  lock_images();
  if (!valid_token(token)) { unlock_images(); return -1; }
  entry = find_image(token, NULL);
  if (entry == NULL || entry->state != IMAGE_READY) {
    unlock_images(); return -2;
  }
  hash_image(entry->vector + sizeof(int64_t), (size_t)entry->length, actual);
  if (memcmp(actual, entry->digest, 32u) != 0) {
    wipe(actual, sizeof(actual)); unlock_images(); return -3;
  }
  wipe(actual, sizeof(actual));
  entry->state = IMAGE_WRITING;
  unlock_images();
#ifdef ET_C2_SAVE_IMAGE_RECEIPT_TESTING
  if (image_writer_hook != NULL) image_writer_hook(token);
#endif
  status = et_checkpoint_io_atomic_write_v1(path, entry->vector,
                                             entry->length, 0);
  if (status < 0) abort(); /* The C1 writer has only zero/positive statuses. */
  lock_images();
  prior = NULL;
  if (find_image(token, &prior) != entry || entry->state != IMAGE_WRITING) abort();
  free_image(entry);
  if (status == 0) {
    entry->state = IMAGE_PUBLISHED;
  } else {
    unlink_image(entry, prior);
    wipe(entry, sizeof(*entry));
    free(entry);
  }
  unlock_images();
  return status;
}

int64_t et_c2_private_save_image_release_v1(int64_t token) {
  image_owner *entry, *prior = NULL;
  if (token <= 0) return -1;
  lock_images();
  if (!valid_token(token)) { unlock_images(); return -1; }
  entry = find_image(token, &prior);
  if (entry == NULL || entry->state == IMAGE_WRITING) {
    unlock_images(); return -2;
  }
  unlink_image(entry, prior);
  free_image(entry);
  wipe(entry, sizeof(*entry));
  free(entry);
  unlock_images();
  return 0;
}

#ifdef ET_C2_SAVE_IMAGE_RECEIPT_TESTING
uint64_t et_c2_private_save_image_live_bytes_v1(void) {
  uint64_t total = 0u;
  image_owner *entry;
  lock_images();
  for (entry = image_live; entry != NULL; entry = entry->next)
    total += entry->vector != NULL ? entry->length : 0u;
  unlock_images();
  return total;
}
void et_c2_private_save_image_test_set_issued_v1(uint64_t issued) {
  lock_images();
  if (image_live != NULL || issued > (uint64_t)INT64_MAX) abort();
  image_issued = issued;
  unlock_images();
}
void et_c2_private_save_image_test_corrupt_v1(int64_t token, uint64_t index) {
  image_owner *entry;
  lock_images();
  entry = find_image(token, NULL);
  if (entry == NULL || entry->state != IMAGE_READY || index >= entry->length)
    abort();
  entry->vector[sizeof(int64_t) + (size_t)index] ^= 1u;
  unlock_images();
}
void et_c2_private_save_image_test_writer_hook_v1(void (*hook)(int64_t)) {
  image_writer_hook = hook;
}
#endif
#endif
