#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

typedef ssize_t (*write_fn)(int, const void *, size_t);

ssize_t write(int descriptor, const void *buffer, size_t length) {
  static write_fn real_write;
  static unsigned int stdout_calls;
  const char *mode;

  if (real_write == NULL) {
    void *symbol = dlsym(RTLD_NEXT, "write");
    memcpy(&real_write, &symbol, sizeof(real_write));
    if (real_write == NULL) {
      errno = EIO;
      return -1;
    }
  }
  mode = getenv("CLI3_TEST_WRITE_MODE");
  if (descriptor != STDOUT_FILENO || mode == NULL) {
    return real_write(descriptor, buffer, length);
  }
  if (strcmp(mode, "short-once") == 0 && stdout_calls++ == 0u && length > 5u) {
    return real_write(descriptor, buffer, 5u);
  }
  if (strcmp(mode, "error-after-prefix") == 0) {
    if (stdout_calls++ == 0u && length > 7u) {
      return real_write(descriptor, buffer, 7u);
    }
    errno = EIO;
    return -1;
  }
  return real_write(descriptor, buffer, length);
}
