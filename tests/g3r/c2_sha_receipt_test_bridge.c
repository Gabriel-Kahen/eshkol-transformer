#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int path_from_bytevector(const void *opaque, char path[4097]) {
  int64_t length;
  if (opaque == NULL) return 0;
  memcpy(&length, opaque, sizeof(length));
  if (length <= 0 || length > 4096) return 0;
  const char *bytes = (const char *)opaque + sizeof(length);
  if (memchr(bytes, 0, (size_t)length) != NULL) return 0;
  memcpy(path, bytes, (size_t)length);
  path[length] = '\0';
  return 1;
}

int64_t et_g3r_c2_receipt_test_replace_v1(const void *source,
                                           const void *destination) {
  char from[4097], to[4097];
  if (!path_from_bytevector(source, from) ||
      !path_from_bytevector(destination, to))
    return 1;
  return rename(from, to) == 0 ? 0 : 2;
}
