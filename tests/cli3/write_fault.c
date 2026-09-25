#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

typedef ssize_t (*write_fn)(int, const void *, size_t);
typedef int (*fsync_fn)(int);

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

int fsync(int descriptor) {
  static fsync_fn real_fsync;
  static int regular_failed;
  static int directory_failed;
  struct stat metadata;
  const char *mode;

  if (real_fsync == NULL) {
    void *symbol = dlsym(RTLD_NEXT, "fsync");
    memcpy(&real_fsync, &symbol, sizeof(real_fsync));
    if (real_fsync == NULL) {
      errno = EIO;
      return -1;
    }
  }
  mode = getenv("CLI3_TEST_FSYNC_MODE");
  if (mode == NULL || fstat(descriptor, &metadata) != 0) {
    return real_fsync(descriptor);
  }
  if (strcmp(mode, "regular-once") == 0 && S_ISREG(metadata.st_mode) &&
      !regular_failed) {
    regular_failed = 1;
    errno = EIO;
    return -1;
  }
  if (strcmp(mode, "directory-once") == 0 && S_ISDIR(metadata.st_mode) &&
      !directory_failed) {
    directory_failed = 1;
    errno = EIO;
    return -1;
  }
  return real_fsync(descriptor);
}
