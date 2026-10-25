#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int calls;
int64_t et_cpu4_private_c4_p3_resume_prefix_v1(
    void *context, void *output, const int64_t ids[3],
    const int64_t rng[4], float logits[256]) {
  assert(context == (void *)&calls && output == (void *)&calls);
  assert(ids[0] == 3 && ids[1] == 197 && ids[2] == 3);
  assert(rng[0] == 1 && rng[1] == 41 && rng[2] == 0 && rng[3] == 0);
  logits[0] = 41.0f;
  ++calls;
  return 0;
}
#include "../../native/cpu4_c4_p3_resume_ffi.c"

typedef struct { int64_t length; int64_t payload[3]; } ids_box;
typedef struct { int64_t length; int64_t payload[4]; } rng_box;
typedef struct { int64_t length; float payload[256]; } logits_box;

int main(void) {
  ids_box ids = {24, {3, 197, 3}};
  rng_box rng = {32, {1, 41, 0, 0}};
  logits_box logits = {1024, {0}};
  assert(et_cpu4_private_c4_p3_resume_eshkol_v1(
      &calls, &calls, &ids, &rng, &logits) == 0);
  assert(calls == 1 && ids.length == 24 && rng.length == 32 &&
         logits.length == 1024 && logits.payload[0] == 41.0f);
  ids.length = 23;
  assert(et_cpu4_private_c4_p3_resume_eshkol_v1(
      &calls, &calls, &ids, &rng, &logits) != 0);
  ids.length = 24;
  rng.length = 31;
  assert(et_cpu4_private_c4_p3_resume_eshkol_v1(
      &calls, &calls, &ids, &rng, &logits) != 0);
  rng.length = 32;
  logits.length = 1023;
  assert(et_cpu4_private_c4_p3_resume_eshkol_v1(
      &calls, &calls, &ids, &rng, &logits) != 0);
  logits.length = 1024;
  assert(et_cpu4_private_c4_p3_resume_eshkol_v1(
      &calls, &calls, NULL, &rng, &logits) != 0);
  assert(calls == 1 && logits.payload[0] == 41.0f);
  puts("CPU4 C4 P3 Eshkol resume ABI PASS: 5 checks");
  return 0;
}
