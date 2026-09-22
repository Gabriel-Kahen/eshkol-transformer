/* Source-private replacement TU: one inherited I2 registry, no consumer authority. */
#include "m3t_f32_integration.c"
#include "m3_call_pins.h"

#define M3_CALL_PIN_COUNT 14u
#define M3_CALL_FULL_MASK UINT16_C(0x3fff)
static const char m3_call_dtype[] = "f32";
static const char m3_call_device[] = "cpu";
static const uint64_t m3_call_shapes[14][2] = {
  {4,4}, {4,4}, {4,4}, {4,4}, {4,8}, {8,4}, {4,0},
  {4,0}, {4,0}, {4,0}, {256,4}, {4,0}, {4,0}, {2,4}
};

typedef enum m3_call_failure {
  M3_CALL_OK, M3_CALL_NULL, M3_CALL_BUFFER, M3_CALL_HANDLE,
  M3_CALL_IDENTITY, M3_CALL_SHAPE, M3_CALL_ACTIVE
} m3_call_failure;
typedef struct m3_call_span { const void *pointer; size_t bytes; } m3_call_span;

#ifdef ET_M3_CALL_TESTING
/* Included-TU tests only. No symbol, callback or extra helper in normal builds. */
static size_t m3_call_test_fail_after = SIZE_MAX;
static unsigned m3_call_test_release_order[14];
static size_t m3_call_test_release_count;
#endif

static int m3_call_view_idle(const et_kernel_tensor_view_v1 *v) {
  return !v->struct_size && !v->data && !v->byte_length && !v->dtype &&
      !v->device && !v->layout && !v->offset_bytes && !v->rank && !v->shape;
}

static int m3_call_idle(const et_g3t_model_pins_internal *pins) {
  if (pins->self || pins->held_mask) return 0;
  for (size_t i = 0; i < M3_CALL_PIN_COUNT; ++i)
    if (pins->parameters[i] || pins->identities[i] || pins->values[i] ||
        !m3_call_view_idle(&pins->views[i])) return 0;
  return 1;
}

static int m3_call_foreign_storage(const void *p, size_t bytes, const void *own);

static int m3_call_storage(const void *p, size_t bytes, size_t alignment) {
  return pointer_span_fits(p, bytes) && aligned_pointer(p, alignment) &&
      !m3_call_foreign_storage(p, bytes, NULL);
}

/* The fixed caller owns accessible storage. These checks reject arithmetic,
 * alignment and overlap defects; they are not arbitrary-pointer admission. */
static int m3_call_error_safe(const et_f32_tensor_error *error,
    const et_g3t_model_pins_internal *pins,
    et_f32_parameter *const *parameters, const void *const *identities) {
  return m3_call_storage(error, sizeof(*error), _Alignof(et_f32_tensor_error)) &&
      !ranges_overlap(error, sizeof(*error), pins, pins ? sizeof(*pins) : 0) &&
      !ranges_overlap(error, sizeof(*error), parameters,
                      parameters ? 14 * sizeof(*parameters) : 0) &&
      !ranges_overlap(error, sizeof(*error), identities,
                      identities ? 14 * sizeof(*identities) : 0);
}

static int32_t m3_call_report(m3_call_failure failure,
    et_f32_tensor_error *error, const char *operation) {
  static const et_f32_tensor_error_category categories[] = {
    ET_F32_TENSOR_ERROR_NONE, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
    ET_F32_TENSOR_ERROR_INVALID_ARGUMENT, ET_F32_TENSOR_ERROR_INVALID_STATE,
    ET_F32_TENSOR_ERROR_INVALID_ARGUMENT, ET_F32_TENSOR_ERROR_SHAPE_MISMATCH,
    ET_F32_TENSOR_ERROR_INVALID_STATE
  };
  static const et_f32_tensor_error_code codes[] = {
    ET_F32_TENSOR_CODE_OK, ET_F32_TENSOR_CODE_NULL_ARGUMENT,
    ET_F32_TENSOR_CODE_INVALID_BUFFER, ET_F32_TENSOR_CODE_INVALID_HANDLE,
    ET_F32_TENSOR_CODE_INVALID_HANDLE, ET_F32_TENSOR_CODE_INVALID_SHAPE,
    ET_F32_TENSOR_CODE_ACTIVE_BORROW
  };
  static const char *const messages[] = {
    "", "required pin operand is null", "pin storage spans are invalid or overlap",
    "pin owner or invariant is invalid", "pin identity is mismatched or duplicate",
    "parameter geometry does not match fixed14", "pin object or owner is active"
  };
  memset(error, 0, sizeof(*error));
  if (failure) {
    error->category = categories[failure];
    error->code = codes[failure];
    memcpy(error->operation, operation, strlen(operation));
    memcpy(error->message, messages[failure], strlen(messages[failure]));
  }
  return (int32_t)categories[failure];
}

/* A corrupted admitted operand must not impersonate unrelated live storage.
 * Excluding its own registry object is safe because the fixed14 span list checks
 * all of that object's control/shape/stride/data spans against one another. */
static int m3_call_array_overlap(const void *p, size_t bytes, const void *array,
    size_t count, size_t size) {
  return count > SIZE_MAX / size || ranges_overlap(p, bytes, array, count * size);
}

static int m3_call_foreign_storage(const void *p, size_t bytes, const void *own) {
  for (const et_f32_tensor *t = live_tensors; t; t = t->registry_next) {
    if (t == own) continue;
    if (ranges_overlap(p, bytes, t, sizeof(*t)) ||
        m3_call_array_overlap(p, bytes, t->shape, t->rank, sizeof(*t->shape)) ||
        m3_call_array_overlap(p, bytes, t->strides, t->rank, sizeof(*t->strides)) ||
        ranges_overlap(p, bytes, t->data, t->byte_length)) return 1;
  }
  for (const et_f32_parameter *v = live_parameters; v; v = v->registry_next)
    if (v != own && ranges_overlap(p, bytes, v, sizeof(*v))) return 1;
  for (const et_f32_tensor_borrow *v = live_borrows; v; v = v->registry_next)
    if (ranges_overlap(p, bytes, v, sizeof(*v))) return 1;
  for (const et_f32_tensor_copy_plan *v = live_copy_plans; v; v = v->registry_next)
    if (ranges_overlap(p, bytes, v, sizeof(*v)) ||
        m3_call_array_overlap(p, bytes, v->assignments, v->count, sizeof(*v->assignments))) return 1;
  for (const et_f32_gradient_plan *v = live_gradient_plans; v; v = v->registry_next) {
    if (ranges_overlap(p, bytes, v, sizeof(*v)) ||
        m3_call_array_overlap(p, bytes, v->entries, v->count, sizeof(*v->entries))) return 1;
    for (size_t i = 0; i < v->count; ++i) {
      const et_f32_parameter *parameter = find_parameter(v->entries[i].parameter);
      const et_f32_tensor *gradient = parameter ? find_tensor(parameter->gradient) : NULL;
      if (!gradient || ranges_overlap(p, bytes, v->entries[i].prepared, gradient->byte_length)) return 1;
    }
  }
  for (const et_f32_gradient_reset_plan *v = live_reset_plans; v; v = v->registry_next)
    if (ranges_overlap(p, bytes, v, sizeof(*v)) ||
        m3_call_array_overlap(p, bytes, v->parameters, v->count, sizeof(*v->parameters))) return 1;
  for (const et_f32_tensor *v = retired_tensors; v; v = v->registry_next)
    if (ranges_overlap(p, bytes, v, sizeof(*v))) return 1;
  for (const et_f32_parameter *v = retired_parameters; v; v = v->registry_next)
    if (ranges_overlap(p, bytes, v, sizeof(*v))) return 1;
  for (const et_f32_tensor_borrow *v = retired_borrows; v; v = v->registry_next)
    if (ranges_overlap(p, bytes, v, sizeof(*v))) return 1;
  for (const et_f32_tensor_copy_plan *v = retired_copy_plans; v; v = v->registry_next)
    if (ranges_overlap(p, bytes, v, sizeof(*v))) return 1;
  for (const et_f32_gradient_plan *v = retired_gradient_plans; v; v = v->registry_next)
    if (ranges_overlap(p, bytes, v, sizeof(*v))) return 1;
  for (const et_f32_gradient_reset_plan *v = retired_reset_plans; v; v = v->registry_next)
    if (ranges_overlap(p, bytes, v, sizeof(*v))) return 1;
  return 0;
}

static int m3_call_add_span(m3_call_span spans[130], size_t *count,
    const void *pointer, size_t bytes, size_t alignment) {
  if (!pointer_span_fits(pointer, bytes) || !aligned_pointer(pointer, alignment))
    return 0;
  for (size_t i = 0; i < *count; ++i)
    if (ranges_overlap(pointer, bytes, spans[i].pointer, spans[i].bytes)) return 0;
  spans[(*count)++] = (m3_call_span){pointer, bytes};
  return 1;
}

/* Every owner is registry-admitted before any field read. Descriptor validation
 * compares fixed string addresses, never untrusted string contents. count is
 * private: 0 for begin preflight, 14 for FULL, a prefix only during rollback. */
static m3_call_failure m3_call_preflight(
    et_f32_parameter *const parameters[14], const void *const identities[14],
    const et_g3t_model_pins_internal *pins, const et_f32_tensor_error *error,
    size_t count, int snapshots) {
  m3_call_span spans[130]; /* 14 * (parameter + 2 * 4 tensor spans) + 4 */
  size_t span_count = 0;
  et_f32_tensor *values[14];
  if (!m3_call_add_span(spans, &span_count, pins, sizeof(*pins),
                        _Alignof(et_g3t_model_pins_internal))) return M3_CALL_BUFFER;
  if (error && !m3_call_add_span(spans, &span_count, error, sizeof(*error),
                                _Alignof(et_f32_tensor_error))) return M3_CALL_BUFFER;
  if (!snapshots &&
      (!m3_call_add_span(spans, &span_count, parameters, 14 * sizeof(*parameters),
                         _Alignof(et_f32_parameter *)) ||
       !m3_call_add_span(spans, &span_count, identities, 14 * sizeof(*identities),
                         _Alignof(const void *)))) return M3_CALL_BUFFER;
  for (size_t i = 0; i < M3_CALL_PIN_COUNT; ++i) {
    et_f32_parameter *p;
    et_f32_tensor *v, *g;
    if (!parameters[i] || !identities[i]) return snapshots ? M3_CALL_HANDLE : M3_CALL_NULL;
    p = find_parameter(parameters[i]);
    if (!p || p->magic != ET_F32_PARAMETER_MAGIC) return M3_CALL_HANDLE;
    v = find_tensor(p->value);
    g = find_tensor(p->gradient);
    if (!v || !g || v->magic != ET_F32_TENSOR_MAGIC ||
        g->magic != ET_F32_TENSOR_MAGIC) return M3_CALL_HANDLE;
    if (p->identity != identities[i]) return snapshots ? M3_CALL_HANDLE : M3_CALL_IDENTITY;
    for (size_t j = 0; j < i; ++j)
      if (parameters[j] == p || identities[j] == identities[i] || values[j] == v)
        return snapshots ? M3_CALL_HANDLE : M3_CALL_IDENTITY;
    values[i] = v;
    if (snapshots && pins->values[i] != v) return M3_CALL_HANDLE;
    if (m3_call_foreign_storage(p, sizeof(*p), p) ||
        !m3_call_add_span(spans, &span_count, p, sizeof(*p), _Alignof(et_f32_parameter)))
      return M3_CALL_BUFFER;
    const size_t rank = m3_call_shapes[i][1] ? 2 : 1;
    const size_t elements = (size_t)m3_call_shapes[i][0] *
        (rank == 2 ? (size_t)m3_call_shapes[i][1] : 1);
    et_f32_tensor *tensors[2] = {v, g};
    for (size_t k = 0; k < 2; ++k) {
      const et_f32_tensor *t = tensors[k];
      if (t->rank != rank || t->element_count != elements ||
          t->byte_length != elements * sizeof(float))
        return snapshots ? M3_CALL_HANDLE : M3_CALL_SHAPE;
      if (m3_call_foreign_storage(t, sizeof(*t), t) ||
          m3_call_foreign_storage(t->shape, rank * sizeof(*t->shape), t) ||
          m3_call_foreign_storage(t->strides, rank * sizeof(*t->strides), t) ||
          m3_call_foreign_storage(t->data, t->byte_length, t) ||
          !m3_call_add_span(spans, &span_count, t, sizeof(*t), _Alignof(et_f32_tensor)) ||
          !m3_call_add_span(spans, &span_count, t->shape, rank * sizeof(*t->shape),
                            _Alignof(uint64_t)) ||
          !m3_call_add_span(spans, &span_count, t->strides, rank * sizeof(*t->strides),
                            _Alignof(size_t)) ||
          !m3_call_add_span(spans, &span_count, t->data, t->byte_length, _Alignof(float)))
        return M3_CALL_BUFFER;
      if (t->shape[0] != m3_call_shapes[i][0] ||
          (rank == 2 && t->shape[1] != m3_call_shapes[i][1]) ||
          t->strides[rank - 1] != sizeof(float) ||
          (rank == 2 && t->strides[0] != m3_call_shapes[i][1] * sizeof(float)))
        return snapshots ? M3_CALL_HANDLE : M3_CALL_SHAPE;
    }
    et_f32_tensor_borrow *expected = i < count
        ? (et_f32_tensor_borrow *)(void *)&pins->views[i] : NULL;
    if (p->plan_pins != (size_t)(i < count) || v->active_borrow != expected ||
        g->active_borrow || v->plan_pins || g->plan_pins)
      return snapshots ? M3_CALL_HANDLE : M3_CALL_ACTIVE;
    if (snapshots) {
      const et_kernel_tensor_view_v1 *view = &pins->views[i];
      if (view->struct_size != sizeof(*view) || view->data != v->data ||
          view->byte_length != v->byte_length || view->dtype != m3_call_dtype ||
          view->device != m3_call_device || view->layout != ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR ||
          view->offset_bytes || view->rank != rank || view->shape != v->shape)
        return M3_CALL_HANDLE;
    }
  }
  return M3_CALL_OK;
}

/* Called only after complete admission. No allocator, mapper or fallible operation
 * occurs after the first release write, including private prefix rollback. */
static void m3_call_drain(et_g3t_model_pins_internal *pins, size_t count) {
  while (count) {
    size_t i = --count;
    pins->values[i]->active_borrow = NULL;
    pins->parameters[i]->plan_pins = 0;
    pins->held_mask &= (uint16_t)~(UINT16_C(1) << i);
#ifdef ET_M3_CALL_TESTING
    m3_call_test_release_order[m3_call_test_release_count++] = (unsigned)i;
#endif
  }
  memset(pins, 0, sizeof(*pins));
}

int32_t et_g3t_model_pins_begin_internal(
    et_f32_parameter *const parameters[14], const void *const identities[14],
    et_g3t_model_pins_internal *pins, et_f32_tensor_error *error) {
  const char *operation = "m3-call-pins-begin";
  if (!m3_call_error_safe(error, pins, parameters, identities))
    return ET_F32_TENSOR_ERROR_INVALID_ARGUMENT;
  if (!pins || !parameters || !identities)
    return m3_call_report(M3_CALL_NULL, error, operation);
  if (!m3_call_storage(pins, sizeof(*pins), _Alignof(et_g3t_model_pins_internal)) ||
      !m3_call_storage(parameters, 14 * sizeof(*parameters), _Alignof(et_f32_parameter *)) ||
      !m3_call_storage(identities, 14 * sizeof(*identities), _Alignof(const void *)) ||
      ranges_overlap(pins, sizeof(*pins), parameters, 14 * sizeof(*parameters)) ||
      ranges_overlap(pins, sizeof(*pins), identities, 14 * sizeof(*identities)) ||
      ranges_overlap(parameters, 14 * sizeof(*parameters), identities, 14 * sizeof(*identities)))
    return m3_call_report(M3_CALL_BUFFER, error, operation);
  if (!m3_call_idle(pins))
    return m3_call_report(pins->self && pins->self != pins ? M3_CALL_HANDLE : M3_CALL_ACTIVE,
                          error, operation);
  m3_call_failure failure = m3_call_preflight(parameters, identities, pins, error, 0, 0);
  if (failure) return m3_call_report(failure, error, operation);
  pins->self = pins;
  for (size_t i = 0; i < M3_CALL_PIN_COUNT; ++i) {
    et_f32_tensor *value = parameters[i]->value;
    pins->parameters[i] = parameters[i];
    pins->identities[i] = identities[i];
    pins->values[i] = value;
    pins->views[i] = (et_kernel_tensor_view_v1){sizeof(pins->views[i]), value->data,
      value->byte_length, m3_call_dtype, m3_call_device,
      ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR, 0, value->rank, value->shape};
  }
#ifdef ET_M3_CALL_TESTING
  m3_call_test_release_count = 0;
#endif
  for (size_t i = 0; i < M3_CALL_PIN_COUNT; ++i) {
#ifdef ET_M3_CALL_TESTING
    if (i == m3_call_test_fail_after) {
      /* Inject only between indivisible acquisitions; preserve the first error. */
      if (pins->self != pins || pins->held_mask != (uint16_t)((1u << i) - 1u) ||
          m3_call_preflight(pins->parameters, pins->identities, pins, error, i, 1)) abort();
      /* Error storage has already been admitted and remains untouched by drain. */
      int32_t result = set_error(error, ET_F32_TENSOR_ERROR_INTERNAL,
          ET_F32_TENSOR_CODE_ALLOCATION_FAILED, operation, "test acquisition failure");
      m3_call_drain(pins, i);
      return result;
    }
#endif
    pins->parameters[i]->plan_pins = 1;
    pins->values[i]->active_borrow = (et_f32_tensor_borrow *)(void *)&pins->views[i];
    pins->held_mask |= (uint16_t)(UINT16_C(1) << i);
  }
  return m3_call_report(M3_CALL_OK, error, operation);
}

int32_t et_g3t_model_pins_check_internal(
    const et_g3t_model_pins_internal *pins, et_f32_tensor_error *error) {
  const char *operation = "m3-call-pins-check";
  if (!m3_call_error_safe(error, pins, NULL, NULL))
    return ET_F32_TENSOR_ERROR_INVALID_ARGUMENT;
  if (!pins) return m3_call_report(M3_CALL_NULL, error, operation);
  if (!m3_call_storage(pins, sizeof(*pins), _Alignof(et_g3t_model_pins_internal)))
    return m3_call_report(M3_CALL_BUFFER, error, operation);
  if (pins->self != pins || pins->held_mask != M3_CALL_FULL_MASK)
    return m3_call_report(M3_CALL_HANDLE, error, operation);
  return m3_call_report(m3_call_preflight(pins->parameters, pins->identities,
      pins, error, 14, 1), error, operation);
}

void et_g3t_model_pins_end_internal(et_g3t_model_pins_internal *pins) {
  if (!m3_call_storage(pins, sizeof(*pins), _Alignof(et_g3t_model_pins_internal))) abort();
  if (m3_call_idle(pins)) return;
  if (pins->self != pins || pins->held_mask != M3_CALL_FULL_MASK ||
      m3_call_preflight(pins->parameters, pins->identities, pins, NULL, 14, 1)) abort();
#ifdef ET_M3_CALL_TESTING
  m3_call_test_release_count = 0;
#endif
  m3_call_drain(pins, 14);
}
