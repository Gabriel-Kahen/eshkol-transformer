/* Independent binary32 replay of the ordered raw D2/E3 observations captured
 * at et_tr3_step_metrics_accumulate_v1 for the six-byte CLI3-B test corpus.
 * This file deliberately does not include or call the production reducer. */
#include <fenv.h>
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF
_Static_assert(sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24 &&
                   FLT_EVAL_METHOD == 0,
               "binary32 reference requires physical f32 evaluation");

static float unpack(uint32_t word) {
  float value;
  memcpy(&value, &word, sizeof(value));
  return value;
}

static uint32_t pack(float value) {
  uint32_t word;
  memcpy(&word, &value, sizeof(word));
  return word;
}

static uint32_t rounded_add(uint32_t left, uint32_t right) {
  volatile float result = unpack(left) + unpack(right);
  return pack(result);
}

static uint32_t rounded_divide(uint32_t numerator, uint32_t weight) {
  volatile float result = unpack(numerator) / unpack(weight);
  return pack(result);
}

static uint32_t replay(const uint32_t numerator[4], const uint32_t weight[4],
                       uint32_t *first, uint32_t *second,
                       uint32_t *total_weight) {
  uint32_t whole_num = 0, whole_weight = 0, update_num = 0, update_weight = 0;
  for (unsigned index = 0; index < 4; ++index) {
    whole_num = rounded_add(whole_num, numerator[index]);
    whole_weight = rounded_add(whole_weight, weight[index]);
    update_num = rounded_add(update_num, numerator[index]);
    update_weight = rounded_add(update_weight, weight[index]);
    if (index == 1) {
      if (update_weight != UINT32_C(0x40800000)) return 0;
      *first = rounded_divide(update_num, update_weight);
      update_num = update_weight = 0;
    }
  }
  if (update_weight != UINT32_C(0x40400000)) return 0;
  *second = rounded_divide(update_num, update_weight);
  *total_weight = whole_weight;
  return rounded_divide(whole_num, whole_weight);
}

int main(void) {
  /* Numerator, weight, mean triples in order from observations-gdb.log. */
  const uint32_t numerator[4] = {UINT32_C(0x4131e51a), UINT32_C(0x41324586),
                                 UINT32_C(0x40b15af9), UINT32_C(0x41278517)};
  const uint32_t weight[4] = {UINT32_C(0x40000000), UINT32_C(0x40000000),
                              UINT32_C(0x3f800000), UINT32_C(0x40000000)};
  const uint32_t observation_mean[4] = {
      UINT32_C(0x40b1e51a), UINT32_C(0x40b24586),
      UINT32_C(0x40b15af9), UINT32_C(0x40a78517)};
  uint32_t first, second, total_weight;
  if (fegetround() != FE_TONEAREST) return 1;
  for (unsigned index = 0; index < 4; ++index) {
    if (rounded_divide(numerator[index], weight[index]) !=
        observation_mean[index]) return 2;
  }
  const uint32_t whole = replay(numerator, weight, &first, &second,
                                &total_weight);
  if (first != UINT32_C(0x40b21550) || second != UINT32_C(0x40aacc63) ||
      total_weight != UINT32_C(0x40e00000) ||
      whole != UINT32_C(0x40aef60f)) return 3;
  uint32_t changed[4];
  memcpy(changed, numerator, sizeof(changed));
  changed[3] ^= UINT32_C(0x00010000);
  uint32_t mutant_first, mutant_second, mutant_weight;
  const uint32_t mutant = replay(changed, weight, &mutant_first,
                                 &mutant_second, &mutant_weight);
  if (mutant == whole || mutant_first != first || mutant_weight != total_weight)
    return 4;
  printf("first=%08x second=%08x total-weight=%08x whole=%08x mutant=%08x\n",
         first, second, total_weight, whole, mutant);
  return 0;
}
