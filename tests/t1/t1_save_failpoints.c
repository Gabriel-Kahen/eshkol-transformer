#define _GNU_SOURCE

#include <errno.h>
#include <stdint.h>
#include <stddef.h>
#include <sys/syscall.h>
#include <unistd.h>

#if !defined(__linux__) || !defined(SYS_write) || !defined(SYS_fsync) ||      \
    !defined(SYS_close) || !defined(SYS_renameat) || !defined(SYS_unlinkat)
#error "T1 save failpoints require the reviewed Linux syscall surface"
#endif

enum et_t1_test_mode {
  ET_T1_TEST_DISABLED = 0,
  ET_T1_TEST_SHORT_WRITE = 1,
  ET_T1_TEST_ZERO_WRITE = 2,
  ET_T1_TEST_EINTR_WRITE = 3,
  ET_T1_TEST_SYNC_TEMP = 4,
  ET_T1_TEST_CLOSE_TEMP = 5,
  ET_T1_TEST_PUBLISH = 6,
  ET_T1_TEST_SYNC_DIRECTORY = 7,
  ET_T1_TEST_CLOSE_DIRECTORY = 8,
  ET_T1_TEST_CLEANUP = 9
};

static struct {
  int64_t mode;
  int64_t injections;
  int64_t writes;
  int64_t syncs;
  int64_t closes;
  int64_t publishes;
  int64_t cleanups;
} et_t1_test_state;

int64_t et_t1_test_io_reset_v1(void) {
  et_t1_test_state.mode = ET_T1_TEST_DISABLED;
  et_t1_test_state.injections = 0;
  et_t1_test_state.writes = 0;
  et_t1_test_state.syncs = 0;
  et_t1_test_state.closes = 0;
  et_t1_test_state.publishes = 0;
  et_t1_test_state.cleanups = 0;
  return 0;
}

int64_t et_t1_test_io_configure_v1(int64_t mode) {
  (void)et_t1_test_io_reset_v1();
  if (mode < ET_T1_TEST_SHORT_WRITE || mode > ET_T1_TEST_CLEANUP) {
    return -1;
  }
  et_t1_test_state.mode = mode;
  return 0;
}

int64_t et_t1_test_io_injections_v1(void) {
  return et_t1_test_state.injections;
}

int64_t et_t1_test_io_writes_v1(void) {
  return et_t1_test_state.writes;
}

ssize_t write(int descriptor, const void *buffer, size_t length) {
  if (et_t1_test_state.mode != ET_T1_TEST_DISABLED) {
    ++et_t1_test_state.writes;
    if (et_t1_test_state.mode == ET_T1_TEST_SHORT_WRITE && length > 7u) {
      ++et_t1_test_state.injections;
      length = 7u;
    } else if (et_t1_test_state.mode == ET_T1_TEST_ZERO_WRITE) {
      ++et_t1_test_state.injections;
      return 0;
    } else if (et_t1_test_state.mode == ET_T1_TEST_EINTR_WRITE &&
               et_t1_test_state.writes == 1) {
      ++et_t1_test_state.injections;
      errno = EINTR;
      return -1;
    }
  }
  return (ssize_t)syscall(SYS_write, descriptor, buffer, length);
}

int fsync(int descriptor) {
  if (et_t1_test_state.mode != ET_T1_TEST_DISABLED) {
    ++et_t1_test_state.syncs;
    if ((et_t1_test_state.mode == ET_T1_TEST_SYNC_TEMP &&
         et_t1_test_state.syncs == 1) ||
        (et_t1_test_state.mode == ET_T1_TEST_SYNC_DIRECTORY &&
         et_t1_test_state.syncs == 2)) {
      ++et_t1_test_state.injections;
      errno = EIO;
      return -1;
    }
  }
  return (int)syscall(SYS_fsync, descriptor);
}

int close(int descriptor) {
  int result;
  if (et_t1_test_state.mode != ET_T1_TEST_DISABLED) {
    ++et_t1_test_state.closes;
    if ((et_t1_test_state.mode == ET_T1_TEST_CLOSE_TEMP &&
         et_t1_test_state.closes == 1) ||
        (et_t1_test_state.mode == ET_T1_TEST_CLOSE_DIRECTORY &&
         et_t1_test_state.closes == 2)) {
      ++et_t1_test_state.injections;
      result = (int)syscall(SYS_close, descriptor);
      (void)result;
      errno = EIO;
      return -1;
    }
  }
  return (int)syscall(SYS_close, descriptor);
}

int renameat(int old_directory, const char *old_name, int new_directory,
             const char *new_name) {
  if (et_t1_test_state.mode != ET_T1_TEST_DISABLED) {
    ++et_t1_test_state.publishes;
    if (et_t1_test_state.mode == ET_T1_TEST_PUBLISH ||
        et_t1_test_state.mode == ET_T1_TEST_CLEANUP) {
      ++et_t1_test_state.injections;
      errno = EIO;
      return -1;
    }
  }
  return (int)syscall(SYS_renameat, old_directory, old_name, new_directory,
                      new_name);
}

int unlinkat(int directory, const char *name, int flags) {
  if (et_t1_test_state.mode != ET_T1_TEST_DISABLED) {
    ++et_t1_test_state.cleanups;
    if (et_t1_test_state.mode == ET_T1_TEST_CLEANUP) {
      ++et_t1_test_state.injections;
      errno = EACCES;
      return -1;
    }
  }
  return (int)syscall(SYS_unlinkat, directory, name, flags);
}
