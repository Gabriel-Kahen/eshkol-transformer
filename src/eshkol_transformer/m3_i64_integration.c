/* M3 replaces the ordinary I1 TU exactly once; standalone I1 stays unchanged. */
#include "../../native/i64_tensor.c"
#include "m3_model.h"

int32_t et_m3_private_i64_unborrowed_v1(const et_i64_tensor *candidate,
    et_i64_tensor_error *error) {
  int32_t rc = preflight_error_operand(error, NULL, 0);
  if (rc) return rc; /* No diagnostic write may follow alias rejection. */
  const et_i64_tensor *tensor = live_tensors;
  while (tensor && tensor != candidate) tensor = tensor->registry_next;
  if (!tensor || tensor->magic != ET_I64_TENSOR_MAGIC)
    return set_i64_error(error, ET_I64_TENSOR_ERROR_INVALID_STATE,
        ET_I64_TENSOR_CODE_INVALID_HANDLE, "m3-i64-unborrowed",
        "tensor is not registered and live");
  if (tensor->active_borrow)
    return set_i64_error(error, ET_I64_TENSOR_ERROR_INVALID_STATE,
        ET_I64_TENSOR_CODE_ACTIVE_BORROW, "m3-i64-unborrowed",
        "tensor has an active borrow");
  return success(error);
}

#ifdef ET_M3_TESTING
int64_t et_m3_test_i64_counts_v1(int64_t field) {
  size_t counts[5] = {0};
  for (const et_i64_tensor *t = live_tensors; t; t = t->registry_next) {
    ++counts[0]; counts[2] += t->byte_length;
    counts[3] += sizeof(*t); counts[4] += t->rank * (sizeof(uint64_t) + sizeof(size_t));
  }
  for (const et_i64_tensor_borrow *b = live_borrows; b; b = b->registry_next) {
    ++counts[1]; counts[3] += sizeof(*b);
  }
  return field >= 0 && field < 5 ? (int64_t)counts[field] : -1;
}
#endif
