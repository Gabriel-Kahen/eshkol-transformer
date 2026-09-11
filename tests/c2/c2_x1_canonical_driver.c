#include "c2_x1_canonical.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned char *read_file(const char *path, size_t *size) {
  FILE *file = fopen(path, "rb");
  long length;
  unsigned char *bytes;
  if (!file || fseek(file, 0, SEEK_END) || (length = ftell(file)) < 0 ||
      fseek(file, 0, SEEK_SET))
    return NULL;
  bytes = (unsigned char *)malloc((size_t)length + 1u);
  if (!bytes || fread(bytes, 1u, (size_t)length, file) != (size_t)length ||
      fclose(file)) {
    free(bytes);
    return NULL;
  }
  *size = (size_t)length;
  return bytes;
}

int main(int argc, char **argv) {
  unsigned char *canonical, *fingerprint;
  size_t canonical_bytes, fingerprint_bytes;
  et_c2_x1_projection_v1 projection;
  et_c2_x1_error_v1 error;
  int32_t status;
  if (argc != 3)
    return 64;
  memset(&projection, 0, sizeof(projection));
  memset(&error, 0, sizeof(error));
  projection.struct_size = sizeof(projection);
  error.struct_size = sizeof(error);
  canonical = read_file(argv[1], &canonical_bytes);
  fingerprint = read_file(argv[2], &fingerprint_bytes);
  if (!canonical || !fingerprint)
    return 65;
  status = et_c2_private_x1_canonical_inspect_v1(canonical, canonical_bytes,
                                                 fingerprint, fingerprint_bytes,
                                                 &projection, &error);
  printf("%d,%u,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n", (int)status,
         error.code, (unsigned long long)projection.vocabulary_size,
         (unsigned long long)projection.context_length,
         (unsigned long long)projection.hidden_size,
         (unsigned long long)projection.layer_count,
         (unsigned long long)projection.query_head_count,
         (unsigned long long)projection.kv_head_count,
         (unsigned long long)projection.head_size,
         (unsigned long long)projection.seed,
         (unsigned long long)projection.accumulation_steps);
  free(canonical);
  free(fingerprint);
  return 0;
}
