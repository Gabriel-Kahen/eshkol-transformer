#include "tr3_public_step_metrics.h"

#include <stdint.h>
#include <string.h>

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
  an = read_word(acc);
  aw = read_word(acc + 4);
  on = read_word(obs);
  ow = read_word(obs + 4);
  if (!finite_nonnegative(an) || !finite_nonnegative(aw) ||
      !finite_nonnegative(on) || !finite_nonnegative(ow) ||
      (ow != UINT32_C(0x3f800000) && ow != UINT32_C(0x40000000)))
    return -1;
  /* The volatile stores make each sum a physical binary32 rounding boundary. */
  volatile float numerator = unpack(an) + unpack(on);
  volatile float weight = unpack(aw) + unpack(ow);
  nn = pack(numerator);
  nw = pack(weight);
  if (!finite_nonnegative(nn) || !finite_nonnegative(nw) || nw == 0) return -1;
  write_word(acc, nn);
  write_word(acc + 4, nw);
  return 0;
}

int64_t et_tr3_step_metrics_mean_bits_v1(void *accumulator) {
  unsigned char *acc = payload(accumulator, 8);
  uint32_t numerator, weight, mean_bits;
  if (!acc) return -1;
  numerator = read_word(acc);
  weight = read_word(acc + 4);
  if (!finite_nonnegative(numerator) || !finite_nonnegative(weight) ||
      weight == 0) return -1;
  volatile float mean = unpack(numerator) / unpack(weight);
  mean_bits = pack(mean);
  if (!finite_nonnegative(mean_bits)) return -1;
  return mean_bits;
}
