#include "eshkol_transformer/mp4_loss_scaler_record.h"

#include <errno.h>
#include <fenv.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>

#pragma STDC FENV_ACCESS ON

#define CHECK(condition)                                                        \
  do {                                                                          \
    if (!(condition)) {                                                         \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition);             \
      exit(EXIT_FAILURE);                                                       \
    }                                                                           \
  } while (0)

enum { RECORD_BYTES = ET_MP4_LOSS_SCALER_RECORD_V1_BYTES };

/* Prefix: Python struct.pack("<8s8I", ...). Trailer: zlib.crc32(prefix). */
static const uint8_t golden[RECORD_BYTES] = {
    0x45, 0x54, 0x4d, 0x50, 0x34, 0x4c, 0x53, 0x00, 0x01, 0x00, 0x00,
    0x00, 0x21, 0x43, 0x05, 0x41, 0x45, 0x23, 0x01, 0x40, 0x10, 0x20,
    0x40, 0x00, 0x04, 0x03, 0x02, 0x01, 0x03, 0x02, 0x01, 0x00, 0x04,
    0x03, 0x02, 0x00, 0xff, 0xff, 0x7f, 0x7f, 0x7f, 0xb3, 0xc1, 0x54};

static float float_of(uint32_t word) {
  float value;
  memcpy(&value, &word, sizeof(value));
  return value;
}

static et_mp4_loss_scaler_v1 fixture(void) {
  et_mp4_loss_scaler_v1 state;
  CHECK(et_mp4_loss_scaler_init(&state, float_of(0x41054321u),
                                 float_of(0x40012345u), float_of(0x00402010u),
                                 0x01020304u, float_of(0x00020304u),
                                 float_of(0x7f7fffffu)) ==
        ET_MP4_LOSS_SCALER_STEP_ACCEPTED);
  state.clean_steps = 0x00010203u;
  return state;
}

static et_mp4_loss_scaler_v1 new_scaler(void) {
  et_mp4_loss_scaler_v1 state;
  CHECK(et_mp4_loss_scaler_init(&state, 8.0f, 3.0f, 0.5f, 2u, 2.0f, 40.0f) ==
        ET_MP4_LOSS_SCALER_STEP_ACCEPTED);
  return state;
}

static void set_word(uint8_t *record, size_t offset, uint32_t word) {
  for (size_t i = 0; i < sizeof(word); ++i)
    record[offset + i] = (uint8_t)(word >> (8u * i));
}

static void test_record_round_trip(void) {
  const et_mp4_loss_scaler_v1 state = fixture();
  et_mp4_loss_scaler_v1 restored = new_scaler();
  _Alignas(uint32_t) uint8_t storage[RECORD_BYTES + 2u];
  uint8_t *record = storage + 1u;
  memset(storage, 0xa5, sizeof(storage));
  CHECK(et_mp4_loss_scaler_encode_v1(&state, record, RECORD_BYTES) ==
        ET_MP4_LOSS_SCALER_RECORD_OK);
  CHECK(memcmp(record, golden, RECORD_BYTES) == 0);
  CHECK(storage[0] == 0xa5u && storage[RECORD_BYTES + 1u] == 0xa5u);
  CHECK(et_mp4_loss_scaler_decode_v1(record, RECORD_BYTES, &restored) ==
        ET_MP4_LOSS_SCALER_RECORD_OK);
  CHECK(memcmp(&state, &restored, sizeof(state)) == 0);
  CHECK(memcmp(record, golden, RECORD_BYTES) == 0);
}

static void test_continuation(void) {
  const struct {
    uint8_t overflow;
    et_mp4_loss_scaler_status outcome;
    float scale;
    uint32_t clean_steps;
  } steps[] = {
      {0u, ET_MP4_LOSS_SCALER_STEP_ACCEPTED, 24.0f, 0u},
      {1u, ET_MP4_LOSS_SCALER_OVERFLOW_BACKOFF, 12.0f, 0u},
      {0u, ET_MP4_LOSS_SCALER_STEP_ACCEPTED, 12.0f, 1u},
      {0u, ET_MP4_LOSS_SCALER_STEP_ACCEPTED, 36.0f, 0u},
      {0u, ET_MP4_LOSS_SCALER_STEP_ACCEPTED, 36.0f, 1u},
      {0u, ET_MP4_LOSS_SCALER_STEP_ACCEPTED, 40.0f, 0u},
      {1u, ET_MP4_LOSS_SCALER_OVERFLOW_BACKOFF, 20.0f, 0u},
      {1u, ET_MP4_LOSS_SCALER_OVERFLOW_BACKOFF, 10.0f, 0u},
      {1u, ET_MP4_LOSS_SCALER_OVERFLOW_BACKOFF, 5.0f, 0u},
      {1u, ET_MP4_LOSS_SCALER_OVERFLOW_BACKOFF, 2.5f, 0u},
      {0u, ET_MP4_LOSS_SCALER_STEP_ACCEPTED, 2.5f, 1u},
      {1u, ET_MP4_LOSS_SCALER_UNDERFLOW_FAULT, 2.5f, 1u},
      {0u, ET_MP4_LOSS_SCALER_STEP_ACCEPTED, 7.5f, 0u},
  };
  et_mp4_loss_scaler_v1 continuous = new_scaler();
  et_mp4_loss_scaler_v1 resumed = fixture();
  uint8_t saved[RECORD_BYTES];
  CHECK(et_mp4_loss_scaler_update(&continuous, 0u) ==
        ET_MP4_LOSS_SCALER_STEP_ACCEPTED);
  CHECK(continuous.clean_steps == 1u);
  CHECK(et_mp4_loss_scaler_encode_v1(&continuous, saved, sizeof(saved)) ==
        ET_MP4_LOSS_SCALER_RECORD_OK);
  CHECK(et_mp4_loss_scaler_decode_v1(saved, sizeof(saved), &resumed) ==
        ET_MP4_LOSS_SCALER_RECORD_OK);
  CHECK(memcmp(&continuous, &resumed, sizeof(continuous)) == 0);
  for (size_t i = 0; i < sizeof(steps) / sizeof(*steps); ++i) {
    CHECK(et_mp4_loss_scaler_update(&continuous, steps[i].overflow) ==
          steps[i].outcome);
    CHECK(et_mp4_loss_scaler_update(&resumed, steps[i].overflow) ==
          steps[i].outcome);
    CHECK(continuous.scale == steps[i].scale &&
          continuous.clean_steps == steps[i].clean_steps);
    CHECK(memcmp(&continuous, &resumed, sizeof(continuous)) == 0);
  }
}

static void expect_bad_record(const uint8_t *record, size_t bytes,
                              et_mp4_loss_scaler_record_status expected) {
  const et_mp4_loss_scaler_v1 before = new_scaler();
  et_mp4_loss_scaler_v1 current = before;
  CHECK(et_mp4_loss_scaler_decode_v1(record, bytes, &current) == expected);
  CHECK(memcmp(&current, &before, sizeof(current)) == 0);
}

static void test_invalid_records(void) {
  uint8_t record[RECORD_BYTES + 1u];
  memcpy(record, golden, RECORD_BYTES);
  record[RECORD_BYTES] = 0u;
  expect_bad_record(NULL, RECORD_BYTES,
                    ET_MP4_LOSS_SCALER_RECORD_INVALID_ARGUMENT);
  for (size_t size = 0; size < RECORD_BYTES; ++size)
    expect_bad_record(record, size, ET_MP4_LOSS_SCALER_RECORD_INVALID_ARGUMENT);
  expect_bad_record(record, sizeof(record),
                    ET_MP4_LOSS_SCALER_RECORD_INVALID_ARGUMENT);
  CHECK(et_mp4_loss_scaler_decode_v1(record, RECORD_BYTES, NULL) ==
        ET_MP4_LOSS_SCALER_RECORD_INVALID_ARGUMENT);
  for (size_t bit = 0; bit < 8u * RECORD_BYTES; ++bit) {
    const uint8_t mask = (uint8_t)(1u << (bit % 8u));
    record[bit / 8u] ^= mask;
    expect_bad_record(record, RECORD_BYTES,
                      ET_MP4_LOSS_SCALER_RECORD_INVALID_RECORD);
    record[bit / 8u] ^= mask;
  }
  record[0] = 'X';
  set_word(record, 40u, 0xa590e0e6u);
  expect_bad_record(record, RECORD_BYTES, ET_MP4_LOSS_SCALER_RECORD_INVALID_RECORD);

  /* Each mutation has an independently computed, valid zlib CRC. */
  const struct {
    size_t offset;
    uint32_t word;
    uint32_t crc;
  } invalid[] = {
      {8u, 0x00000000u, 0xa51bb6d5u},  /* stale version */
      {8u, 0x00000002u, 0x9ddebbc0u},  /* unknown future version */
      {12u, 0x7f800001u, 0x43ecf8e4u}, /* signaling NaN scale */
      {12u, 0x00000000u, 0xa1d350aau}, /* below min_scale */
      {16u, 0x3f800000u, 0xd22357f3u}, /* growth_factor == 1 */
      {20u, 0x3f800000u, 0x791560fdu}, /* backoff_factor == 1 */
      {24u, 0x00000000u, 0xad7588a9u}, /* zero interval */
      {28u, 0x01020304u, 0x1d9b2a70u}, /* clean_steps == growth_interval */
      {32u, 0x80000000u, 0x1d4db681u}, /* negative zero minimum */
      {36u, 0x7f800000u, 0xa7787c11u}, /* infinite maximum */
      {32u, 0x42000000u, 0x8e980d3au}, /* minimum exceeds scale */
      {36u, 0x40000000u, 0x2a9dc967u}, /* maximum below scale */
  };
  for (size_t i = 0; i < sizeof(invalid) / sizeof(*invalid); ++i) {
    memcpy(record, golden, RECORD_BYTES);
    set_word(record, invalid[i].offset, invalid[i].word);
    set_word(record, 40u, invalid[i].crc);
    expect_bad_record(record, RECORD_BYTES,
                      invalid[i].offset == 8u
                          ? ET_MP4_LOSS_SCALER_RECORD_UNSUPPORTED_VERSION
                          : ET_MP4_LOSS_SCALER_RECORD_INVALID_RECORD);
  }
}

static void test_invalid_encode_and_overlap(void) {
  struct {
    et_mp4_loss_scaler_v1 state;
    uint8_t record[RECORD_BYTES + 1u];
  } storage;
  memset(&storage, 0xa5, sizeof(storage));
  storage.state = fixture();
  storage.state.clean_steps = storage.state.growth_interval;
  uint8_t before[sizeof(storage)];
  memcpy(before, &storage, sizeof(before));
  CHECK(et_mp4_loss_scaler_encode_v1(&storage.state, storage.record,
                                     RECORD_BYTES) ==
        ET_MP4_LOSS_SCALER_RECORD_INVALID_ARGUMENT);
  CHECK(memcmp(before, &storage, sizeof(storage)) == 0);

  storage.state = fixture();
  memcpy(before, &storage, sizeof(before));
  CHECK(et_mp4_loss_scaler_encode_v1(NULL, storage.record, RECORD_BYTES) ==
        ET_MP4_LOSS_SCALER_RECORD_INVALID_ARGUMENT);
  CHECK(et_mp4_loss_scaler_encode_v1(&storage.state, NULL, RECORD_BYTES) ==
        ET_MP4_LOSS_SCALER_RECORD_INVALID_ARGUMENT);
  CHECK(et_mp4_loss_scaler_encode_v1(&storage.state, storage.record,
                                     sizeof(storage.record)) ==
        ET_MP4_LOSS_SCALER_RECORD_INVALID_ARGUMENT);
  CHECK(et_mp4_loss_scaler_encode_v1(&storage.state, storage.record,
                                     RECORD_BYTES - 1u) ==
        ET_MP4_LOSS_SCALER_RECORD_INVALID_ARGUMENT);
  CHECK(et_mp4_loss_scaler_encode_v1(&storage.state, (uint8_t *)&storage + 1u,
                                     RECORD_BYTES) ==
        ET_MP4_LOSS_SCALER_RECORD_INVALID_ARGUMENT);
  CHECK(memcmp(before, &storage, sizeof(storage)) == 0);

  memcpy(&storage, golden, RECORD_BYTES);
  memcpy(before, &storage, sizeof(before));
  CHECK(et_mp4_loss_scaler_decode_v1((const uint8_t *)&storage, RECORD_BYTES,
                                     &storage.state) ==
        ET_MP4_LOSS_SCALER_RECORD_INVALID_ARGUMENT);
  CHECK(memcmp(before, &storage, sizeof(storage)) == 0);
}

static void test_environment(void) {
  uint8_t record[RECORD_BYTES];
  const et_mp4_loss_scaler_v1 input = fixture();
  et_mp4_loss_scaler_v1 output = new_scaler();
  fenv_t original;
  CHECK(fegetenv(&original) == 0 && fesetround(FE_UPWARD) == 0);
  /* Enable FTZ and DAZ and start with the inexact sticky flag set. */
  const unsigned int hostile_mxcsr = _mm_getcsr() | 0x8040u | 0x0020u;
  _mm_setcsr(hostile_mxcsr);
  errno = EDOM;

  CHECK(et_mp4_loss_scaler_encode_v1(&input, record, RECORD_BYTES) ==
        ET_MP4_LOSS_SCALER_RECORD_OK);
  CHECK(et_mp4_loss_scaler_decode_v1(record, RECORD_BYTES, &output) ==
        ET_MP4_LOSS_SCALER_RECORD_OK);
  CHECK(memcmp(record, golden, RECORD_BYTES) == 0);
  CHECK(memcmp(&input, &output, sizeof(input)) == 0);
  CHECK(et_mp4_loss_scaler_update(&output, 0u) ==
        ET_MP4_LOSS_SCALER_UNSUPPORTED_ENVIRONMENT);
  CHECK(memcmp(&input, &output, sizeof(input)) == 0);

  set_word(record, 12u, 0x7f800001u);
  set_word(record, 40u, 0x43ecf8e4u);
  CHECK(et_mp4_loss_scaler_decode_v1(record, RECORD_BYTES, &output) ==
        ET_MP4_LOSS_SCALER_RECORD_INVALID_RECORD);
  CHECK(memcmp(&input, &output, sizeof(input)) == 0);
  CHECK(errno == EDOM && _mm_getcsr() == hostile_mxcsr &&
        fegetround() == FE_UPWARD);
  CHECK(fesetenv(&original) == 0);
}

int main(void) {
  test_record_round_trip();
  test_continuation();
  test_invalid_records();
  test_invalid_encode_and_overlap();
  test_environment();
  puts("MP4 loss-scaler v1 record, rejected inputs, and continuation: PASS");
  return 0;
}
