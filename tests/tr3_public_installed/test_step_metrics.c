#include "tr3_public_step_metrics.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int64_t length; unsigned char bytes[12]; } buffer;

static void put(unsigned char *out, uint32_t word) {
  for (unsigned i = 0; i < 4; ++i) out[i] = (unsigned char)(word >> (8u * i));
}

static uint32_t get(const unsigned char *in) {
  return (uint32_t)in[0] | ((uint32_t)in[1] << 8) |
         ((uint32_t)in[2] << 16) | ((uint32_t)in[3] << 24);
}

static void check(int yes) { if (!yes) abort(); }

int main(void) {
  buffer acc = {.length = 8};
  buffer obs = {.length = 12};
  put(obs.bytes, UINT32_C(0x3fc00000)); /* 1.5 */
  put(obs.bytes + 4, UINT32_C(0x40000000)); /* weight 2 */
  check(et_tr3_step_metrics_accumulate_v1(&acc, &obs) == 0);
  put(obs.bytes, UINT32_C(0x3e800000)); /* 0.25 */
  put(obs.bytes + 4, UINT32_C(0x3f800000)); /* weight 1 */
  check(et_tr3_step_metrics_accumulate_v1(&acc, &obs) == 0);
  check(get(acc.bytes) == UINT32_C(0x3fe00000));
  check(get(acc.bytes + 4) == UINT32_C(0x40400000));
  check(et_tr3_step_metrics_mean_bits_v1(&acc) == INT64_C(0x3f155555));

  buffer before = acc;
  put(obs.bytes, UINT32_C(0x7fc00000));
  check(et_tr3_step_metrics_accumulate_v1(&acc, &obs) < 0);
  check(memcmp(&before, &acc, sizeof(acc)) == 0);
  put(obs.bytes, UINT32_C(0x3f800000));
  put(obs.bytes + 4, UINT32_C(0x00000000));
  check(et_tr3_step_metrics_accumulate_v1(&acc, &obs) < 0);
  check(memcmp(&before, &acc, sizeof(acc)) == 0);
  acc.length = 7;
  check(et_tr3_step_metrics_mean_bits_v1(&acc) < 0);
  puts("TR3-PUBLIC-STEP-METRICS-KERNEL-PASS");
  return 0;
}
