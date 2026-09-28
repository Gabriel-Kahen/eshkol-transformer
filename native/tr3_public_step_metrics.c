#include "tr3_public_step_metrics.h"

#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <stdint.h>
#include <string.h>
#if !defined(__x86_64__)
#error "TR3 public step requires the reviewed x86-64 binary32 lane"
#endif
#include <xmmintrin.h>

#if defined(__clang__)
#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF
#endif
_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4 && FLT_RADIX == 2 &&
                   FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
                   FLT_EVAL_METHOD == 0,
               "TR3 public step requires IEEE binary32 evaluation");

static int fp_supported(void) {
  unsigned short control;
  __asm__ volatile("fnstcw %0" : "=m"(control));
  const unsigned mxcsr = _mm_getcsr();
  return (control & 0x003fu) == 0x003fu && (control & 0x0c00u) == 0 &&
         (mxcsr & 0x1f80u) == 0x1f80u && (mxcsr & 0xe040u) == 0;
}

static uint32_t read_word(const unsigned char *source) {
  return (uint32_t)source[0] | ((uint32_t)source[1] << 8) |
         ((uint32_t)source[2] << 16) | ((uint32_t)source[3] << 24);
}

static void write_word(unsigned char *target, uint32_t word) {
  for (unsigned i = 0; i < 4; ++i) target[i] = (unsigned char)(word >> (8u * i));
}

static int finite_nonnegative(uint32_t bits) {
  return (bits & UINT32_C(0x80000000)) == 0 &&
         (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

static float unpack(uint32_t bits) {
  float value;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

static uint32_t pack(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static unsigned char *payload(void *object, int64_t expected) {
  int64_t length;
  if (!object) return NULL;
  memcpy(&length, object, sizeof(length));
  if (length != expected) return NULL;
  return (unsigned char *)object + sizeof(length);
}

int64_t et_tr3_step_metrics_accumulate_v1(void *accumulator, void *observation) {
  unsigned char *acc = payload(accumulator, 8);
  unsigned char *obs = payload(observation, 12);
  uint32_t an, aw, on, ow, nn, nw;
  if (!acc || !obs) return -1;
  if (!fp_supported()) return -2;
  an = read_word(acc);
  aw = read_word(acc + 4);
  on = read_word(obs);
  ow = read_word(obs + 4);
  if (!finite_nonnegative(an) || !finite_nonnegative(aw) ||
      !finite_nonnegative(on) || !finite_nonnegative(ow) ||
      (ow != UINT32_C(0x3f800000) && ow != UINT32_C(0x40000000)))
    return -1;
  fenv_t environment;
  if (fegetenv(&environment) != 0) return -2;
  /* The volatile stores make each sum a physical binary32 rounding boundary. */
  volatile float numerator = unpack(an) + unpack(on);
  volatile float weight = unpack(aw) + unpack(ow);
  nn = pack(numerator);
  nw = pack(weight);
  if (fesetenv(&environment) != 0) return -2;
  if (!finite_nonnegative(nn) || !finite_nonnegative(nw) || nw == 0) return -1;
  write_word(acc, nn);
  write_word(acc + 4, nw);
  return 0;
}

int64_t et_tr3_step_metrics_mean_bits_v1(void *accumulator) {
  unsigned char *acc = payload(accumulator, 8);
  uint32_t numerator, weight, mean_bits;
  if (!acc) return -1;
  if (!fp_supported()) return -2;
  numerator = read_word(acc);
  weight = read_word(acc + 4);
  if (!finite_nonnegative(numerator) || !finite_nonnegative(weight) ||
      weight == 0) return -1;
  fenv_t environment;
  if (fegetenv(&environment) != 0) return -2;
  volatile float mean = unpack(numerator) / unpack(weight);
  mean_bits = pack(mean);
  if (fesetenv(&environment) != 0) return -2;
  if (!finite_nonnegative(mean_bits)) return -1;
  return mean_bits;
}
