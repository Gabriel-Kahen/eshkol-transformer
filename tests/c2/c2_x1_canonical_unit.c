#include "c2_x1_canonical.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct test_bytevector {
  int64_t length;
  uint8_t bytes[1024];
} test_bytevector;

int main(void) {
  static const uint8_t empty[] = "";
  et_c2_x1_projection_v1 projection;
  et_c2_x1_error_v1 error;
  memset(&projection, 0xa5, sizeof(projection));
  projection.struct_size = sizeof(projection);
  error.struct_size = sizeof(error);
  if (et_c2_private_x1_canonical_inspect_v1(empty, 0u, empty, 0u, &projection,
                                            &error) != ET_C2_X1_CORRUPT_DATA ||
      projection.vocabulary_size != UINT64_C(0xa5a5a5a5a5a5a5a5) ||
      error.code != ET_C2_X1_CODE_FINGERPRINT)
    return 1;
  if (et_c2_private_x1_canonical_inspect_v1(NULL, 0u, empty, 0u, &projection,
                                            &error) !=
      ET_C2_X1_INVALID_ARGUMENT)
    return 2;
  projection.struct_size = 0u;
  if (et_c2_private_x1_canonical_inspect_v1(empty, 0u, empty, 0u, &projection,
                                            &error) !=
      ET_C2_X1_INVALID_ARGUMENT)
    return 3;
  projection.struct_size = sizeof(projection);
  error.struct_size = 0u;
  error.category = UINT32_C(0xa5a5a5a5);
  if (et_c2_private_x1_canonical_inspect_v1(empty, 0u, empty, 0u, &projection,
                                            &error) !=
          ET_C2_X1_INVALID_ARGUMENT ||
      error.category != UINT32_C(0xa5a5a5a5))
    return 4;
  error.struct_size = sizeof(error);
  if (et_c2_private_x1_canonical_inspect_v1(
          (const uint8_t *)&projection, sizeof(projection), empty, 0u,
          &projection, &error) != ET_C2_X1_INVALID_ARGUMENT)
    return 5;
  {
    test_bytevector canonical = {0, {0}}, fingerprint = {93, {0}},
                    output = {72, {0}};
    uint8_t before[72];
    memset(output.bytes, 0xa5, 72u);
    memcpy(before, output.bytes, 72u);
    if (et_c2_private_x1_canonical_inspect_bytevectors_v1(
            &canonical, &fingerprint, &output) != ET_C2_X1_CORRUPT_DATA)
      return 6;
    if (memcmp(before, output.bytes, 72u) != 0)
      return 7;
    output.length = 71;
    if (et_c2_private_x1_canonical_inspect_bytevectors_v1(
            &canonical, &fingerprint, &output) != ET_C2_X1_INVALID_ARGUMENT)
      return 8;
    output.length = 72;
    if (et_c2_private_x1_canonical_inspect_bytevectors_v1(
            &canonical, &fingerprint, &canonical) != ET_C2_X1_INVALID_ARGUMENT)
      return 9;
    fingerprint.length = 92;
    if (et_c2_private_x1_canonical_inspect_bytevectors_v1(
            &canonical, &fingerprint, &output) != ET_C2_X1_INVALID_ARGUMENT)
      return 10;
    canonical.length = 16385;
    fingerprint.length = 93;
    if (et_c2_private_x1_canonical_inspect_bytevectors_v1(
            &canonical, &fingerprint, &output) != ET_C2_X1_INVALID_ARGUMENT)
      return 11;
  }
  puts("c2 x1 canonical unit: PASS");
  return 0;
}
