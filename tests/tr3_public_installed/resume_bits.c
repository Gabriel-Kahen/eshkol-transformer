#define _POSIX_C_SOURCE 200809L
#include <eshkol/eshkol.h>
#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <time.h>

/* Test-only, read-only inspection of a public f32 immediate in a one-slot
 * extern box. It adds no authority to the installed aggregate. */
typedef struct resume_box {
  int64_t length;
  eshkol_tagged_value_t value;
} resume_box;

_Static_assert(sizeof(eshkol_tagged_value_t) == 16,
               "pinned Eshkol tagged-value size");
_Static_assert(offsetof(resume_box, value) == 8,
               "pinned Eshkol one-slot vector layout");

int64_t et_tr3_public_resume_f32_bits_v1(void *input) {
  const resume_box *box = (const resume_box *)input;
  eshkol_tagged_value_t value;
  uint32_t bits = 0;
  if (box == NULL || ESHKOL_GET_SUBTYPE(box) != HEAP_SUBTYPE_VECTOR ||
      box->length != 1)
    return -1;
  value = box->value;
  if (eshkol_value_is_f32_v1(&value) != 1 ||
      eshkol_value_f32_to_bits_v1(&value, &bits) != ESHKOL_VALUE_F32_OK)
    return -1;
  return (int64_t)bits;
}

/* Test-only pacing lets the external driver observe the producer's ready
 * marker before this callback exhausts its bounded request wait. */
int64_t et_tr3_public_resume_poll_delay_v1(void) {
  struct timespec remaining = {0, 1000000L};
  while (nanosleep(&remaining, &remaining) != 0) {
    if (errno != EINTR)
      return 0;
  }
  return 1;
}
