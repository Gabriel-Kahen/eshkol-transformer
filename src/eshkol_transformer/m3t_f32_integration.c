/* The M3T aggregate substitutes this TU for native/f32_tensor.c, exactly once.
 * The standalone I2 implementation and its public symbol inventory are unchanged. */
#include "../../native/f32_tensor.c"
#include "m3t_f32_scoped.h"

int32_t et_f32_tensor_scoped_begin_internal(et_f32_tensor *tensor,
    et_f32_scoped_guard_internal *guard, et_f32_tensor_error *error) {
  int32_t rc = preflight_output(guard, guard ? sizeof(*guard) : 0, error,
                               "m3t-scoped-begin");
  if (rc) return rc;
  if (!guard) return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
      ET_F32_TENSOR_CODE_NULL_ARGUMENT, "m3t-scoped-begin", "guard required");
  if (guard->owner || (guard->self && guard->self != guard))
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_STATE,
        ET_F32_TENSOR_CODE_ACTIVE_BORROW, "m3t-scoped-begin", "guard is active or copied");
  rc = require_tensor(tensor, "m3t-scoped-begin", error);
  if (rc) return rc;
  if (tensor->active_borrow || tensor->plan_pins)
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_STATE,
        ET_F32_TENSOR_CODE_ACTIVE_BORROW, "m3t-scoped-begin", "tensor is borrowed or pinned");
  guard->view = (et_kernel_tensor_view_v1){sizeof(guard->view), tensor->data,
    tensor->byte_length, "f32", "cpu", ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR, 0,
    tensor->rank, tensor->shape};
  guard->owner = tensor;
  guard->self = guard;
  tensor->active_borrow = (et_f32_tensor_borrow *)(void *)guard;
  return success(error);
}

int32_t et_f32_tensor_scoped_end_internal(et_f32_scoped_guard_internal *guard) {
  if (!guard || (guard->self && guard->self != guard))
    return ET_F32_TENSOR_ERROR_INVALID_STATE;
  if (!guard->owner) return 0;
  et_f32_tensor *owner = find_tensor(guard->owner);
  if (guard->self != guard || !owner || owner->magic != ET_F32_TENSOR_MAGIC ||
      owner->active_borrow != (et_f32_tensor_borrow *)(void *)guard)
    return ET_F32_TENSOR_ERROR_INVALID_STATE;
  owner->active_borrow = NULL;
  guard->owner = NULL;
  memset(&guard->view, 0, sizeof(guard->view));
  return 0;
}

int32_t et_m3t_f32_parameter_preflight_internal(et_f32_parameter *parameter,
    et_f32_tensor_error *error) {
  int32_t rc = require_parameter(parameter, "m3t-parameter-preflight", error);
  if (rc) return rc;
  if (parameter->plan_pins || parameter->value->active_borrow ||
      parameter->gradient->active_borrow || parameter->value->plan_pins ||
      parameter->gradient->plan_pins)
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_STATE,
      ET_F32_TENSOR_CODE_INVALID_HANDLE, "m3t-parameter-preflight",
      "parameter is borrowed or pinned");
  return success(error);
}
#ifdef ET_F32_TENSOR_TESTING
et_f32_tensor *et_m3t_test_parameter_gradient(et_f32_parameter *p) {
  return find_parameter(p) ? p->gradient : NULL;
}
#endif
