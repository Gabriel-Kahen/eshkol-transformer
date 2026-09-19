#include "c2_checkpoint_format.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_file(const char *path, uint8_t **bytes, size_t *length) {
  FILE *file = fopen(path, "rb");
  long end;
  if (file == NULL || fseek(file, 0, SEEK_END) != 0 ||
      (end = ftell(file)) < 0 || fseek(file, 0, SEEK_SET) != 0) {
    if (file != NULL) fclose(file);
    return 0;
  }
  *length = (size_t)end;
  *bytes = (uint8_t *)malloc(*length == 0u ? 1u : *length);
  if (*bytes == NULL || fread(*bytes, 1u, *length, file) != *length ||
      fclose(file) != 0) {
    free(*bytes); *bytes = NULL; return 0;
  }
  return 1;
}

int main(int argc, char **argv) {
  et_c2_checkpoint_limits_v1 limits = {
      sizeof(limits), ET_C2_WIRE_MAX_FILE_BYTES,
      ET_C2_WIRE_MAX_METADATA_BYTES, ET_C2_WIRE_MAX_TENSOR_BYTES,
      ET_C2_WIRE_MAX_TENSORS, 1u};
  et_c2_checkpoint_view_v1 view = {0};
  et_c2_checkpoint_format_error_v1 error = {0};
  uint8_t *bytes = NULL;
  size_t length = 0u;
  uint32_t mode = ET_C2_FORMAT_INSPECT;
  int32_t status;
  if (argc < 2 || argc > 7 || !read_file(argv[1], &bytes, &length)) {
    fprintf(stderr, "usage/read failure: %s\n", errno ? strerror(errno) : "input");
    return 111;
  }
  if (argc > 2) mode = (uint32_t)strtoul(argv[2], NULL, 10);
  if (argc > 3) limits.enforce_operational_profile = (uint32_t)strtoul(argv[3], NULL, 10);
  if (argc > 4) limits.maximum_tensor_bytes = strtoull(argv[4], NULL, 10);
  if (argc > 5) limits.maximum_tensors = (uint32_t)strtoul(argv[5], NULL, 10);
  if (argc > 6) limits.maximum_metadata_bytes = strtoull(argv[6], NULL, 10);
  view.struct_size = sizeof(view); error.struct_size = sizeof(error);
  status = et_c2_checkpoint_parse_v1(bytes, length, &limits, mode, &view, &error);
  printf("%d %u %llu %u %u %llu %llu\n", status, error.code,
         (unsigned long long)error.offset, view.model_tensor_count,
         view.total_tensor_count, (unsigned long long)view.model_container_bytes,
         (unsigned long long)view.optimizer_payload_bytes);
  free(bytes);
  return status == ET_C2_FORMAT_OK ? 0 : 1;
}
