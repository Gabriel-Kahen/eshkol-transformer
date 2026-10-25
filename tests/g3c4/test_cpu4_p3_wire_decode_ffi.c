#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define ET_CPU4_C4_HISTORY_PRIVATE 1
#define ET_CPU4_C4_P3_WIRE_PRIVATE 1
#include "../../src/eshkol_transformer/cpu4_c4_capture_private.h"

static int calls;
int64_t et_cpu4_private_c4_p3_wire_decode_v1(
    const void *wire, uint64_t wire_bytes, const void *values,
    uint64_t values_bytes, const uint8_t sha[32], const uint8_t fp[96],
    et_cpu4_c4_capture_record_v1 *record) {
  assert(wire_bytes == 400 && values_bytes == 4768);
  assert(((const uint8_t *)wire)[0] == 71 &&
         ((const uint8_t *)values)[0] == 72 && sha[0] == 73 && fp[0] == 74);
  record->size = 224;
  ++calls;
  return 0;
}
#include "../../native/cpu4_c4_p3_wire_decode_ffi.c"

typedef struct { int64_t length; uint8_t data[400]; } wire_box;
typedef struct { int64_t length; uint8_t data[4768]; } values_box;
typedef struct { int64_t length; uint8_t data[32]; } sha_box;
typedef struct { int64_t length; uint8_t data[96]; } fp_box;
typedef struct { int64_t length; et_cpu4_c4_capture_record_v1 data; } record_box;

int main(void) {
  wire_box wire = {400, {71}};
  values_box values = {4768, {72}};
  sha_box sha = {32, {73}};
  fp_box fp = {96, {74}};
  record_box record = {224, {0}};
  assert(et_cpu4_private_c4_p3_wire_decode_eshkol_v1(
      &wire, 400, &values, 4768, &sha, &fp, &record) == 0);
  assert(calls == 1 && record.length == 224 && record.data.size == 224);
  wire.length = 399;
  assert(et_cpu4_private_c4_p3_wire_decode_eshkol_v1(
      &wire, 400, &values, 4768, &sha, &fp, &record) != 0);
  wire.length = 400;
  values.length = 4767;
  assert(et_cpu4_private_c4_p3_wire_decode_eshkol_v1(
      &wire, 400, &values, 4768, &sha, &fp, &record) != 0);
  values.length = 4768;
  sha.length = 31;
  assert(et_cpu4_private_c4_p3_wire_decode_eshkol_v1(
      &wire, 400, &values, 4768, &sha, &fp, &record) != 0);
  sha.length = 32;
  fp.length = 95;
  assert(et_cpu4_private_c4_p3_wire_decode_eshkol_v1(
      &wire, 400, &values, 4768, &sha, &fp, &record) != 0);
  fp.length = 96;
  record.length = 223;
  assert(et_cpu4_private_c4_p3_wire_decode_eshkol_v1(
      &wire, 400, &values, 4768, &sha, &fp, &record) != 0);
  assert(calls == 1 && record.data.size == 224);
  puts("CPU4 C4 P3 Eshkol decode ABI PASS: 6 checks");
  return 0;
}
