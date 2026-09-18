#include "c2_checkpoint_format.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "CHECK failed line %d: %s\n", \
                                         __LINE__, #x); return 1; } } while (0)

int main(void) {
  uint8_t bytes[288] = {0};
  et_c2_checkpoint_limits_v1 limits = {
      sizeof(limits), ET_C2_WIRE_MAX_FILE_BYTES, ET_C2_WIRE_MAX_METADATA_BYTES,
      ET_C2_WIRE_MAX_TENSOR_BYTES, ET_C2_WIRE_MAX_TENSORS, 0u};
  et_c2_checkpoint_view_v1 view = {0};
  et_c2_checkpoint_format_error_v1 error = {0};
  et_c2_checkpoint_view_v1 before;
  int32_t status;
  view.struct_size = sizeof(view);
  error.struct_size = sizeof(error);
  memset((uint8_t *)&view + sizeof(view.struct_size), 0xa5,
         sizeof(view) - sizeof(view.struct_size));
  before = view;
  status = et_c2_checkpoint_parse_v1(bytes, sizeof(bytes), &limits,
                                     ET_C2_FORMAT_INSPECT, &view, &error);
  CHECK(status == ET_C2_FORMAT_CORRUPT_DATA);
  CHECK(memcmp(&view, &before, sizeof(view)) == 0);
  CHECK(et_c2_checkpoint_parse_v1(NULL, 0u, &limits, ET_C2_FORMAT_INSPECT,
                                  &view, &error) == ET_C2_FORMAT_INVALID_ARGUMENT);
  limits.maximum_tensors = ET_C2_WIRE_MAX_TENSORS + 1u;
  CHECK(et_c2_checkpoint_parse_v1(bytes, sizeof(bytes), &limits,
                                  ET_C2_FORMAT_INSPECT, &view, &error) ==
        ET_C2_FORMAT_INVALID_ARGUMENT);
  puts("C2 format unit: PASS");
  return 0;
}
