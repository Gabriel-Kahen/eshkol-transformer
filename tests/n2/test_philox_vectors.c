#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(ET_N2_TESTING)
#error "the native Philox vector test requires ET_N2_TESTING"
#endif

int32_t et_n2_test_philox4x32_10_v1(const int64_t state_words[4],
                                    uint32_t output[4]);

static size_t checks;

#define CHECK(condition)                                                        \
  do {                                                                          \
    checks++;                                                                    \
    if (!(condition)) {                                                          \
      (void)fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,            \
                    #condition);                                                 \
      exit(1);                                                                   \
    }                                                                            \
  } while (0)

static int64_t encode_signed_word(uint64_t value) {
  if (value <= (uint64_t)INT64_MAX) return (int64_t)value;
  return -(int64_t)(UINT64_MAX - value) - INT64_C(1);
}

static uint64_t combine_words(uint32_t low, uint32_t high) {
  return (uint64_t)low | ((uint64_t)high << 32);
}

int main(void) {
  static const uint32_t inputs[][6] = {
      {UINT32_C(0x00000000), UINT32_C(0x00000000),
       UINT32_C(0x00000000), UINT32_C(0x00000000),
       UINT32_C(0x00000000), UINT32_C(0x00000000)},
      {UINT32_C(0xffffffff), UINT32_C(0xffffffff),
       UINT32_C(0xffffffff), UINT32_C(0xffffffff),
       UINT32_C(0xffffffff), UINT32_C(0xffffffff)},
      {UINT32_C(0x243f6a88), UINT32_C(0x85a308d3),
       UINT32_C(0x13198a2e), UINT32_C(0x03707344),
       UINT32_C(0xa4093822), UINT32_C(0x299f31d0)}};
  static const uint32_t expected[][4] = {
      {UINT32_C(0x6627e8d5), UINT32_C(0xe169c58d),
       UINT32_C(0xbc57ac4c), UINT32_C(0x9b00dbd8)},
      {UINT32_C(0x408f276d), UINT32_C(0x41c83b0e),
       UINT32_C(0xa20bc7c6), UINT32_C(0x6d5451fd)},
      {UINT32_C(0xd16cfe09), UINT32_C(0x94fdcceb),
       UINT32_C(0x5001e420), UINT32_C(0x24126ea1)}};

  for (size_t vector = 0; vector < sizeof(inputs) / sizeof(inputs[0]);
       vector++) {
    const uint64_t counter_low =
        combine_words(inputs[vector][0], inputs[vector][1]);
    const uint64_t counter_high =
        combine_words(inputs[vector][2], inputs[vector][3]);
    const uint64_t key = combine_words(inputs[vector][4], inputs[vector][5]);
    int64_t state[] = {INT64_C(1), encode_signed_word(key),
                       encode_signed_word(counter_low),
                       encode_signed_word(counter_high)};
    uint32_t actual[4] = {0u, 0u, 0u, 0u};
    CHECK(et_n2_test_philox4x32_10_v1(state, actual) == 1);
    CHECK(memcmp(actual, expected[vector], sizeof(actual)) == 0);
    CHECK(state[1] == encode_signed_word(key));
    CHECK(state[2] == encode_signed_word(counter_low));
    CHECK(state[3] == encode_signed_word(counter_high));
  }

  {
    int64_t invalid_version[] = {INT64_C(2), INT64_C(-1), INT64_MIN,
                                 INT64_MAX};
    uint32_t unchanged[] = {UINT32_C(0x11111111), UINT32_C(0x22222222),
                            UINT32_C(0x33333333), UINT32_C(0x44444444)};
    uint32_t snapshot[4];
    memcpy(snapshot, unchanged, sizeof(snapshot));
    CHECK(et_n2_test_philox4x32_10_v1(invalid_version, unchanged) == 0);
    CHECK(memcmp(unchanged, snapshot, sizeof(unchanged)) == 0);
    CHECK(et_n2_test_philox4x32_10_v1(NULL, unchanged) == 0);
    CHECK(et_n2_test_philox4x32_10_v1(invalid_version, NULL) == 0);
  }

  (void)printf("N2 native Philox PASS (%zu checks)\n", checks);
  return 0;
}
