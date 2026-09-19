#define _POSIX_C_SOURCE 200809L

#include "eshkol_transformer/kernel_abi.h"

#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

int64_t et_d2_test_view_match_v1(const et_kernel_tensor_view_v1 *view,
                                 int64_t rows, int64_t columns,
                                 int64_t dtype_kind,
                                 const void *expected_bytevector,
                                 int64_t expected_bytes) {
  int64_t declared;
  const uint8_t *expected;
  const char *dtype = dtype_kind == 1 ? "i64" : "bool";
  if (view == NULL || rows <= 0 || columns <= 0 ||
      (dtype_kind != 1 && dtype_kind != 2) ||
      expected_bytevector == NULL || expected_bytes < 0) {
    return 0;
  }
  memcpy(&declared, expected_bytevector, sizeof(declared));
  expected = (const uint8_t *)expected_bytevector + sizeof(declared);
  if (declared != expected_bytes ||
      view->struct_size != ET_KERNEL_TENSOR_VIEW_V1_0_SIZE ||
      view->dtype == NULL || strcmp(view->dtype, dtype) != 0 ||
      view->device == NULL || strcmp(view->device, "cpu") != 0 ||
      view->layout != ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR ||
      view->offset_bytes != 0 || view->rank != 2 || view->shape == NULL ||
      view->shape[0] != (uint64_t)rows ||
      view->shape[1] != (uint64_t)columns ||
      view->byte_length != (uint64_t)expected_bytes ||
      (expected_bytes != 0 && view->data == NULL)) {
    return 0;
  }
  return memcmp(view->data, expected, (size_t)expected_bytes) == 0 ? 1 : 0;
}

int64_t et_d2_test_flip_byte_v1(const char *path, int64_t path_bytes,
                                int64_t offset) {
  int descriptor;
  uint8_t byte;
  if (path == NULL || path_bytes <= 1 || offset < 0 ||
      path[path_bytes - 1] != '\0') {
    return 0;
  }
  descriptor = open(path, O_RDWR);
  if (descriptor < 0 || pread(descriptor, &byte, 1, (off_t)offset) != 1) {
    if (descriptor >= 0) (void)close(descriptor);
    return 0;
  }
  byte ^= UINT8_C(1);
  if (pwrite(descriptor, &byte, 1, (off_t)offset) != 1 ||
      fsync(descriptor) != 0 || close(descriptor) != 0) {
    return 0;
  }
  return 1;
}
