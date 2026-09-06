#include "eshkol_transformer/n2_primitives_abi.h"

#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

#define COUNT(a) (sizeof(a) / sizeof((a)[0]))
#define ANY_ERROR UINT32_MAX
#define ARENA_BYTES 512u

static size_t checks;

#define CHECK(x)                                                               \
  do {                                                                         \
    checks++;                                                                  \
    if (!(x)) {                                                                \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x);             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

typedef struct operation_spec {
  const char *capability;
  const char *operation;
  uint64_t shape[4];
  size_t rank;
} operation_spec;

static const operation_spec operations[] = {
    {"kernel.embedding-forward", "embedding.forward", {1, 4, 3, 2}, 4},
    {"kernel.embedding-backward", "embedding.backward", {1, 4, 3, 2}, 4},
    {"kernel.matmul", "linear.forward", {1, 2, 3, 4}, 4},
    {"kernel.matmul", "linear.forward-no-bias", {1, 2, 3, 4}, 4},
    {"kernel.matmul", "linear.backward", {1, 2, 3, 4}, 4},
    {"kernel.matmul", "linear.backward-no-bias", {1, 2, 3, 4}, 4},
    {"kernel.norm", "layer-norm.forward", {1, 2, 4, 0}, 3},
    {"kernel.norm", "layer-norm.backward", {1, 2, 4, 0}, 3},
    {"kernel.activation", "gelu.forward", {1, 2, 5, 0}, 3},
    {"kernel.activation", "gelu.backward", {1, 2, 5, 0}, 3},
    {"kernel.activation", "relu.forward", {1, 2, 5, 0}, 3},
    {"kernel.activation", "relu.backward", {1, 2, 5, 0}, 3},
    {"kernel.dropout", "dropout.train.forward", {1, 2, 5, 0}, 3},
    {"kernel.dropout", "dropout.train.backward", {1, 2, 5, 0}, 3},
    {"kernel.dropout", "dropout.eval.forward", {1, 2, 5, 0}, 3},
    {"kernel.dropout", "dropout.eval.backward", {1, 2, 5, 0}, 3},
    {"kernel.residual", "residual.forward", {1, 3, 4, 0}, 3},
    {"kernel.residual", "residual.backward", {1, 3, 4, 0}, 3},
};

typedef union aligned_arena {
  max_align_t alignment;
  unsigned char bytes[ARENA_BYTES];
} aligned_arena;

typedef struct call_case {
  et_kernel_request_v1 request;
  et_kernel_call_v1 call;
  et_kernel_tensor_view_v1 inputs[5];
  et_kernel_tensor_view_v1 outputs[4];
  uint64_t request_shape[4];
  uint64_t input_shapes[5][4];
  uint64_t output_shapes[4][4];
  aligned_arena input_data[5];
  aligned_arena output_data[4];
  size_t input_count;
  size_t output_count;
} call_case;

typedef struct backing_snapshot {
  aligned_arena inputs[5];
  aligned_arena outputs[4];
} backing_snapshot;

static const et_kernel_provider_v1 *resolve(void *context,
                                            const char *symbol) {
  (void)context;
  CHECK(strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0);
  return et_n2_kernel_provider_v1();
}

static et_kernel_runtime *new_runtime(void) {
  et_kernel_runtime *runtime = NULL;
  et_kernel_error error;
  CHECK(et_kernel_runtime_discover(resolve, NULL, &runtime, &error) == 0);
  CHECK(runtime != NULL);
  return runtime;
}

static size_t dtype_size(const char *dtype) {
  return strcmp(dtype, "i64") == 0 ? sizeof(int64_t) : sizeof(float);
}

static size_t element_count(size_t rank, const uint64_t *shape) {
  size_t result = 1u;
  for (size_t i = 0; i < rank; i++) {
    CHECK(shape[i] <= SIZE_MAX);
    CHECK(shape[i] == 0u || result <= SIZE_MAX / (size_t)shape[i]);
    result *= (size_t)shape[i];
  }
  return result;
}

static et_kernel_tensor_view_v1 make_view(void *data, const char *dtype,
                                          size_t rank, uint64_t *shape) {
  et_kernel_tensor_view_v1 view = {
      sizeof(view),
      data,
      element_count(rank, shape) * dtype_size(dtype),
      dtype,
      "cpu",
      ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,
      0u,
      rank,
      shape,
  };
  CHECK(view.byte_length <= ARENA_BYTES);
  return view;
}

static void set_shape(uint64_t destination[4], size_t rank, uint64_t a,
                      uint64_t b, uint64_t c, uint64_t d) {
  const uint64_t values[4] = {a, b, c, d};
  CHECK(rank <= 4u);
  memcpy(destination, values, rank * sizeof(values[0]));
}

static void add_input(call_case *c, const char *dtype, size_t rank, uint64_t a,
                      uint64_t b, uint64_t d, uint64_t e) {
  const size_t index = c->input_count++;
  CHECK(index < COUNT(c->inputs));
  set_shape(c->input_shapes[index], rank, a, b, d, e);
  c->inputs[index] = make_view(c->input_data[index].bytes, dtype, rank,
                               c->input_shapes[index]);
}

static void add_output(call_case *c, const char *dtype, size_t rank, uint64_t a,
                       uint64_t b, uint64_t d, uint64_t e) {
  const size_t index = c->output_count++;
  CHECK(index < COUNT(c->outputs));
  set_shape(c->output_shapes[index], rank, a, b, d, e);
  c->outputs[index] = make_view(c->output_data[index].bytes, dtype, rank,
                                c->output_shapes[index]);
}

static int is_operation(const operation_spec *spec, const char *name) {
  return strcmp(spec->operation, name) == 0;
}

static int begins(const operation_spec *spec, const char *prefix) {
  return strncmp(spec->operation, prefix, strlen(prefix)) == 0;
}

static void initialize_payloads(call_case *c) {
  for (size_t i = 0; i < COUNT(c->input_data); i++) {
    float *values = (float *)c->input_data[i].bytes;
    for (size_t j = 0; j < ARENA_BYTES / sizeof(float); j++)
      values[j] = (float)(j % 7u + 1u) * 0.03125f;
  }
  for (size_t i = 0; i < COUNT(c->output_data); i++)
    memset(c->output_data[i].bytes, 0xa5, ARENA_BYTES);
}

static void make_case(call_case *c, const operation_spec *spec) {
  const uint64_t n = spec->shape[0], t = spec->shape[1], a = spec->shape[2],
                 b = spec->rank == 4u ? spec->shape[3] : 0u;
  memset(c, 0, sizeof(*c));
  memcpy(c->request_shape, spec->shape, sizeof(c->request_shape));
  initialize_payloads(c);

  if (begins(spec, "embedding.")) {
    add_input(c, "i64", 2, n, t, 0, 0);
    add_input(c, "f32", is_operation(spec, "embedding.forward") ? 2u : 3u,
              is_operation(spec, "embedding.forward") ? a : n,
              is_operation(spec, "embedding.forward") ? b : t,
              is_operation(spec, "embedding.forward") ? 0 : b, 0);
    add_output(c, "f32", is_operation(spec, "embedding.forward") ? 3u : 2u,
               is_operation(spec, "embedding.forward") ? n : a,
               is_operation(spec, "embedding.forward") ? t : b,
               is_operation(spec, "embedding.forward") ? b : 0, 0);
    {
      int64_t *ids = (int64_t *)c->input_data[0].bytes;
      for (size_t i = 0; i < (size_t)(n * t); i++) ids[i] = (int64_t)(i % a);
    }
  } else if (begins(spec, "linear.")) {
    const int backward = begins(spec, "linear.backward");
    const int bias = is_operation(spec, "linear.forward") ||
                     is_operation(spec, "linear.backward");
    add_input(c, "f32", 3, n, t, a, 0);
    add_input(c, "f32", 2, b, a, 0, 0);
    if (backward)
      add_input(c, "f32", 3, n, t, b, 0);
    else if (bias)
      add_input(c, "f32", 1, b, 0, 0, 0);
    if (backward) {
      add_output(c, "f32", 3, n, t, a, 0);
      add_output(c, "f32", 2, b, a, 0, 0);
      if (bias) add_output(c, "f32", 1, b, 0, 0, 0);
    } else {
      add_output(c, "f32", 3, n, t, b, 0);
    }
  } else if (begins(spec, "layer-norm.")) {
    const int backward = is_operation(spec, "layer-norm.backward");
    add_input(c, "f32", 3, n, t, a, 0);
    add_input(c, "f32", 1, a, 0, 0, 0);
    if (backward) {
      add_input(c, "f32", 0, 0, 0, 0, 0);
      add_input(c, "f32", 3, n, t, a, 0);
    } else {
      add_input(c, "f32", 1, a, 0, 0, 0);
      add_input(c, "f32", 0, 0, 0, 0, 0);
    }
    *(float *)c->inputs[backward ? 2u : 3u].data = 1.0e-5f;
    add_output(c, "f32", 3, n, t, a, 0);
    if (backward) {
      add_output(c, "f32", 1, a, 0, 0, 0);
      add_output(c, "f32", 1, a, 0, 0, 0);
    }
  } else if (begins(spec, "gelu.") || begins(spec, "relu.")) {
    add_input(c, "f32", 3, n, t, a, 0);
    if (strstr(spec->operation, ".backward") != NULL)
      add_input(c, "f32", 3, n, t, a, 0);
    add_output(c, "f32", 3, n, t, a, 0);
  } else if (begins(spec, "dropout.")) {
    const int train = begins(spec, "dropout.train.");
    add_input(c, "f32", 3, n, t, a, 0);
    if (train) {
      add_input(c, "f32", 0, 0, 0, 0, 0);
      add_input(c, "i64", 1, 4, 0, 0, 0);
      *(float *)c->inputs[1].data = 0.25f;
      {
        int64_t state[4] = {1, INT64_C(0x123456789abcdef), 3, 4};
        memcpy(c->inputs[2].data, state, sizeof(state));
      }
    }
    add_output(c, "f32", 3, n, t, a, 0);
    if (is_operation(spec, "dropout.train.forward"))
      add_output(c, "i64", 1, 4, 0, 0, 0);
  } else {
    const int backward = is_operation(spec, "residual.backward");
    add_input(c, "f32", 3, n, t, a, 0);
    if (!backward) add_input(c, "f32", 3, n, t, a, 0);
    add_output(c, "f32", 3, n, t, a, 0);
    if (backward) add_output(c, "f32", 3, n, t, a, 0);
  }

  c->request = (et_kernel_request_v1){sizeof(c->request),
                                      spec->operation,
                                      "f32",
                                      "cpu",
                                      spec->rank,
                                      c->request_shape,
                                      1u,
                                      {0}};
  c->call = (et_kernel_call_v1){sizeof(c->call),
                                spec->capability,
                                &c->request,
                                c->input_count,
                                sizeof(c->inputs[0]),
                                c->input_count * sizeof(c->inputs[0]),
                                c->inputs,
                                c->output_count,
                                sizeof(c->outputs[0]),
                                c->output_count * sizeof(c->outputs[0]),
                                c->outputs};
}

static void snapshot_backing(const call_case *c, backing_snapshot *snapshot) {
  memcpy(snapshot->inputs, c->input_data, sizeof(snapshot->inputs));
  memcpy(snapshot->outputs, c->output_data, sizeof(snapshot->outputs));
}

static void check_backing(const call_case *c,
                          const backing_snapshot *snapshot) {
  CHECK(memcmp(snapshot->inputs, c->input_data, sizeof(snapshot->inputs)) == 0);
  CHECK(memcmp(snapshot->outputs, c->output_data,
               sizeof(snapshot->outputs)) == 0);
}

static void reject_unchanged(et_kernel_runtime *runtime, call_case *c,
                             uint32_t category, uint32_t code,
                             const char *label) {
  backing_snapshot before;
  et_kernel_error error;
  int32_t result;
  snapshot_backing(c, &before);
  result = et_kernel_runtime_dispatch(runtime, &c->call, &error);
  if (result == 0 ||
      (category != ANY_ERROR && error.category != category) ||
      (code != ANY_ERROR && error.code != code)) {
    fprintf(stderr,
            "unexpected result for %s/%s: result=%d category=%u code=%u "
            "operation=%s message=%s\n",
            c->request.operation, label, result, error.category, error.code,
            error.operation, error.message);
    CHECK(0);
  }
  check_backing(c, &before);
}

static void success_inputs_unchanged(et_kernel_runtime *runtime, call_case *c,
                                     const char *label) {
  aligned_arena before[5];
  et_kernel_error error;
  memcpy(before, c->input_data, sizeof(before));
  if (et_kernel_runtime_dispatch(runtime, &c->call, &error) != 0) {
    fprintf(stderr, "unexpected success failure for %s/%s: %u/%u %s\n",
            c->request.operation, label, error.category, error.code,
            error.message);
    CHECK(0);
  }
  CHECK(memcmp(before, c->input_data, sizeof(before)) == 0);
}

static void resize_for_view_shape(et_kernel_tensor_view_v1 *view) {
  view->byte_length = element_count(view->rank, view->shape) *
                      dtype_size(view->dtype);
  CHECK(view->byte_length <= ARENA_BYTES - 16u);
}

static void test_schema_and_views(et_kernel_runtime *runtime,
                                  const operation_spec *spec) {
  call_case c;

#define FRESH() make_case(&c, spec)
#define REJECT(category, code, label)                                          \
  reject_unchanged(runtime, &c, category, code, label)

  FRESH();
  c.call.input_count--;
  c.call.input_bytes = c.call.input_count * c.call.input_stride;
  if (c.call.input_count == 0u) {
    c.call.inputs = NULL;
    c.call.input_stride = 0u;
  }
  REJECT(ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_INVALID_BUFFER,
         "too-few-inputs");

  FRESH();
  c.inputs[c.input_count] = c.inputs[0];
  c.inputs[c.input_count].data = c.input_data[c.input_count].bytes;
  c.call.input_count++;
  c.call.input_bytes = c.call.input_count * c.call.input_stride;
  REJECT(ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_INVALID_BUFFER,
         "too-many-inputs");

  FRESH();
  c.call.output_count--;
  c.call.output_bytes = c.call.output_count * c.call.output_stride;
  if (c.call.output_count == 0u) {
    c.call.outputs = NULL;
    c.call.output_stride = 0u;
  }
  REJECT(ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_INVALID_BUFFER,
         "too-few-outputs");

  FRESH();
  c.outputs[c.output_count] = c.outputs[0];
  c.outputs[c.output_count].data = c.output_data[c.output_count].bytes;
  c.call.output_count++;
  c.call.output_bytes = c.call.output_count * c.call.output_stride;
  REJECT(ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_INVALID_BUFFER,
         "too-many-outputs");

  FRESH();
  c.call.input_bytes--;
  REJECT(ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_INTEGER_OVERFLOW,
         "input-table-span");
  FRESH();
  c.call.output_bytes--;
  REJECT(ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_INTEGER_OVERFLOW,
         "output-table-span");
  FRESH();
  c.call.input_count = ET_KERNEL_MAX_TENSORS + 1u;
  REJECT(ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_INTEGER_OVERFLOW,
         "input-count-overflow");
  FRESH();
  c.call.output_count = ET_KERNEL_MAX_TENSORS + 1u;
  REJECT(ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_INTEGER_OVERFLOW,
         "output-count-overflow");
  FRESH();
  c.call.input_stride = ET_KERNEL_TENSOR_VIEW_V1_0_SIZE - 1u;
  REJECT(ET_KERNEL_ERROR_VERSION_MISMATCH,
         ET_KERNEL_CODE_INVALID_STRUCT_SIZE, "input-stride");
  FRESH();
  c.call.output_stride = ET_KERNEL_TENSOR_VIEW_V1_0_SIZE - 1u;
  REJECT(ET_KERNEL_ERROR_VERSION_MISMATCH,
         ET_KERNEL_CODE_INVALID_STRUCT_SIZE, "output-stride");
  FRESH();
  {
    _Alignas(et_kernel_tensor_view_v1)
        unsigned char raw[sizeof(c.inputs) + 1u];
    memcpy(raw + 1u, c.inputs, sizeof(c.inputs));
    c.call.inputs = raw + 1u;
    REJECT(ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_INVALID_BUFFER,
           "input-table-alignment");
  }
  FRESH();
  {
    _Alignas(et_kernel_tensor_view_v1)
        unsigned char raw[sizeof(c.outputs) + 1u];
    memcpy(raw + 1u, c.outputs, sizeof(c.outputs));
    c.call.outputs = raw + 1u;
    REJECT(ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_INVALID_BUFFER,
           "output-table-alignment");
  }

  FRESH();
  c.request.rank--;
  REJECT(ET_KERNEL_ERROR_UNSUPPORTED,
         ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED, "request-rank");
  FRESH();
  c.request_shape[0] = 0u;
  REJECT(ET_KERNEL_ERROR_UNSUPPORTED,
         ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED, "request-zero-extent");
  FRESH();
  c.request_shape[c.request.rank - 1u] = 7u;
  REJECT(ET_KERNEL_ERROR_UNSUPPORTED,
         ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED, "request-out-of-row");
  FRESH();
  c.request.dtype = "f64";
  REJECT(ET_KERNEL_ERROR_UNSUPPORTED,
         ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED, "request-dtype");
  FRESH();
  c.request.device = "cuda";
  REJECT(ET_KERNEL_ERROR_DEVICE_MISMATCH, ET_KERNEL_CODE_INVALID_BUFFER,
         "request-device");
  FRESH();
  c.request.operation = "n2.unknown";
  REJECT(ET_KERNEL_ERROR_UNSUPPORTED,
         ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED, "request-operation");
  FRESH();
  c.call.capability = "kernel.unknown";
  REJECT(ET_KERNEL_ERROR_UNSUPPORTED,
         ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED, "request-capability");

  for (size_t output = 0; output < 2u; output++) {
    const size_t count = output ? c.output_count : c.input_count;
    for (size_t index = 0; index < count; index++) {
      et_kernel_tensor_view_v1 *view;
      const char *side = output ? "output" : "input";
      (void)side;
      FRESH();
      view = output ? &c.outputs[index] : &c.inputs[index];
      view->struct_size--;
      REJECT(ET_KERNEL_ERROR_VERSION_MISMATCH,
             ET_KERNEL_CODE_INVALID_STRUCT_SIZE, "view-struct-size");
      FRESH();
      view = output ? &c.outputs[index] : &c.inputs[index];
      view->dtype = strcmp(view->dtype, "f32") == 0 ? "i64" : "f32";
      resize_for_view_shape(view);
      REJECT(ET_KERNEL_ERROR_DTYPE_MISMATCH, ET_KERNEL_CODE_INVALID_TEXT,
             "view-dtype");
      FRESH();
      view = output ? &c.outputs[index] : &c.inputs[index];
      view->device = "other";
      REJECT(ET_KERNEL_ERROR_DEVICE_MISMATCH, ET_KERNEL_CODE_INVALID_BUFFER,
             "view-device");
      FRESH();
      view = output ? &c.outputs[index] : &c.inputs[index];
      view->layout = 0u;
      REJECT(ET_KERNEL_ERROR_NONCONTIGUOUS, ET_KERNEL_CODE_INVALID_BUFFER,
             "view-layout");
      FRESH();
      view = output ? &c.outputs[index] : &c.inputs[index];
      view->offset_bytes = 4u;
      REJECT(ET_KERNEL_ERROR_NONCONTIGUOUS, ET_KERNEL_CODE_INVALID_BUFFER,
             "view-offset");
      FRESH();
      view = output ? &c.outputs[index] : &c.inputs[index];
      view->byte_length--;
      REJECT(ET_KERNEL_ERROR_SHAPE_MISMATCH, ET_KERNEL_CODE_INVALID_BUFFER,
             "view-byte-length");
      FRESH();
      view = output ? &c.outputs[index] : &c.inputs[index];
      view->data = (unsigned char *)view->data + 1u;
      REJECT(ET_KERNEL_ERROR_INVALID_ARGUMENT,
             ET_KERNEL_CODE_PROVIDER_REJECTED, "view-data-alignment");
      FRESH();
      view = output ? &c.outputs[index] : &c.inputs[index];
      if (view->rank != 0u) {
        view->rank--;
        resize_for_view_shape(view);
      } else {
        view->rank = 1u;
        view->shape = output ? c.output_shapes[index] : c.input_shapes[index];
        ((uint64_t *)view->shape)[0] = 1u;
        resize_for_view_shape(view);
      }
      REJECT(ET_KERNEL_ERROR_SHAPE_MISMATCH, ET_KERNEL_CODE_INVALID_SHAPE,
             "view-rank");
      FRESH();
      view = output ? &c.outputs[index] : &c.inputs[index];
      if (view->rank != 0u) {
        ((uint64_t *)view->shape)[0] = 0u;
        resize_for_view_shape(view);
        REJECT(ET_KERNEL_ERROR_SHAPE_MISMATCH, ET_KERNEL_CODE_INVALID_SHAPE,
               "view-zero-extent");
      }
      FRESH();
      view = output ? &c.outputs[index] : &c.inputs[index];
      if (view->rank != 0u) {
        ((uint64_t *)view->shape)[view->rank - 1u]++;
        resize_for_view_shape(view);
        REJECT(ET_KERNEL_ERROR_SHAPE_MISMATCH, ET_KERNEL_CODE_INVALID_SHAPE,
               "view-positive-wrong-extent");
      }
      FRESH();
      view = output ? &c.outputs[index] : &c.inputs[index];
      if (view->rank != 0u) {
        _Alignas(uint64_t) unsigned char raw[5u * sizeof(uint64_t)] = {0};
        memcpy(raw + 1u, view->shape, view->rank * sizeof(uint64_t));
        view->shape = (const uint64_t *)(raw + 1u);
        REJECT(ET_KERNEL_ERROR_INVALID_ARGUMENT,
               ET_KERNEL_CODE_INVALID_BUFFER, "view-shape-alignment");
      }
    }
  }

#undef REJECT
#undef FRESH
}

static void test_aliases(et_kernel_runtime *runtime,
                         const operation_spec *spec) {
  call_case c;
#define FRESH() make_case(&c, spec)
#define REJECT(code, label)                                                    \
  reject_unchanged(runtime, &c, ET_KERNEL_ERROR_INVALID_ARGUMENT, code, label)
    FRESH();
    for (size_t left = 0; left < c.input_count; left++) {
      for (size_t right = left + 1u; right < c.input_count; right++) {
        FRESH();
        c.inputs[right].data = c.inputs[left].data;
        if (is_operation(spec, "residual.forward"))
          success_inputs_unchanged(runtime, &c,
                                   "residual-exact-input-alias");
        else
          REJECT(ET_KERNEL_CODE_PROVIDER_REJECTED, "full-input-alias");

        FRESH();
        if (c.inputs[left].byte_length > sizeof(float)) {
          c.inputs[right].data =
              (unsigned char *)c.inputs[left].data + sizeof(float);
          REJECT(ET_KERNEL_CODE_PROVIDER_REJECTED, "partial-input-alias");
        } else if (c.inputs[right].byte_length > sizeof(float)) {
          c.inputs[left].data =
              (unsigned char *)c.inputs[right].data + sizeof(float);
          REJECT(ET_KERNEL_CODE_PROVIDER_REJECTED, "partial-input-alias");
        }
      }
    }

    for (size_t output = 0; output < c.output_count; output++) {
      for (size_t input = 0; input < c.input_count; input++) {
        FRESH();
        c.outputs[output].data = c.inputs[input].data;
        REJECT(ET_KERNEL_CODE_ALIASING_OUTPUT, "exact-output-input-alias");
        if (c.inputs[input].byte_length > sizeof(float)) {
          FRESH();
          c.outputs[output].data =
              (unsigned char *)c.inputs[input].data + sizeof(float);
          REJECT(ET_KERNEL_CODE_ALIASING_OUTPUT,
                 "partial-output-input-alias");
        }
      }
    }
    for (size_t left = 0; left < c.output_count; left++) {
      for (size_t right = left + 1u; right < c.output_count; right++) {
        FRESH();
        c.outputs[right].data = c.outputs[left].data;
        REJECT(ET_KERNEL_CODE_ALIASING_OUTPUT, "exact-output-output-alias");
        if (c.outputs[left].byte_length > sizeof(float)) {
          FRESH();
          c.outputs[right].data =
              (unsigned char *)c.outputs[left].data + sizeof(float);
          REJECT(ET_KERNEL_CODE_ALIASING_OUTPUT,
                 "partial-output-output-alias");
        }
      }
    }
#undef REJECT
#undef FRESH
}

static void test_nonfinite_inputs(et_kernel_runtime *runtime,
                                  const operation_spec *spec) {
  static const uint32_t bad_bits[] = {UINT32_C(0x7fc00001),
                                      UINT32_C(0x7f800000),
                                      UINT32_C(0xff800000)};
  call_case c;
  make_case(&c, spec);
  for (size_t input = 0; input < c.input_count; input++) {
    if (strcmp(c.inputs[input].dtype, "f32") != 0) continue;
    for (size_t bad = 0; bad < COUNT(bad_bits); bad++) {
      make_case(&c, spec);
      memcpy(c.inputs[input].data, &bad_bits[bad], sizeof(uint32_t));
      reject_unchanged(runtime, &c, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                       ET_KERNEL_CODE_PROVIDER_REJECTED, "nonfinite-input");
    }
  }
}

static void test_embedding_ids(et_kernel_runtime *runtime) {
  for (size_t op = 0; op < 2u; op++) {
    call_case c;
    make_case(&c, &operations[op]);
    *(int64_t *)c.inputs[0].data = -1;
    reject_unchanged(runtime, &c, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_PROVIDER_REJECTED, "negative-id");
    make_case(&c, &operations[op]);
    *(int64_t *)c.inputs[0].data = (int64_t)c.request_shape[2];
    reject_unchanged(runtime, &c, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_PROVIDER_REJECTED, "id-at-vocabulary");
  }
}

static void test_epsilon(et_kernel_runtime *runtime) {
  static const uint32_t bits[] = {UINT32_C(0x00000000), UINT32_C(0x80000000),
                                  UINT32_C(0xbf800000), UINT32_C(0x7f800000),
                                  UINT32_C(0xff800000), UINT32_C(0x7fc00001)};
  for (size_t oi = 6u; oi <= 7u; oi++) {
    call_case c;
    const size_t epsilon_input = oi == 6u ? 3u : 2u;
    for (size_t value = 0; value < COUNT(bits); value++) {
      make_case(&c, &operations[oi]);
      memcpy(c.inputs[epsilon_input].data, &bits[value], sizeof(uint32_t));
      reject_unchanged(runtime, &c, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                       ET_KERNEL_CODE_PROVIDER_REJECTED, "invalid-epsilon");
    }
  }
}

static void test_dropout_state(et_kernel_runtime *runtime) {
  static const uint32_t invalid_probability[] = {
      UINT32_C(0xbf000000), UINT32_C(0x3f800000), UINT32_C(0x40000000),
      UINT32_C(0x7f800000), UINT32_C(0xff800000), UINT32_C(0x7fc00001)};
  for (size_t oi = 12u; oi <= 13u; oi++) {
    call_case c;
    for (size_t p = 0; p < COUNT(invalid_probability); p++) {
      make_case(&c, &operations[oi]);
      memcpy(c.inputs[1].data, &invalid_probability[p], sizeof(uint32_t));
      reject_unchanged(runtime, &c, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                       ET_KERNEL_CODE_PROVIDER_REJECTED,
                       "invalid-dropout-probability");
    }
    make_case(&c, &operations[oi]);
    ((int64_t *)c.inputs[2].data)[0] = 2;
    reject_unchanged(runtime, &c, ET_KERNEL_ERROR_VERSION_MISMATCH,
                     ET_KERNEL_CODE_PROVIDER_REJECTED, "unknown-rng-version");
    make_case(&c, &operations[oi]);
    ((int64_t *)c.inputs[2].data)[2] = -1;
    ((int64_t *)c.inputs[2].data)[3] = -1;
    reject_unchanged(runtime, &c, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_PROVIDER_REJECTED, "max-counter-sentinel");

    /* The admitted ten-element row consumes three Philox blocks.  Checked
     * 128-bit preflight must reject max-1 plus three without mutating either
     * forward or backward outputs. */
    make_case(&c, &operations[oi]);
    ((int64_t *)c.inputs[2].data)[2] = -2;
    ((int64_t *)c.inputs[2].data)[3] = -1;
    reject_unchanged(runtime, &c, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_PROVIDER_REJECTED,
                     "multi-block-counter-overflow");

    /* max-3 plus exactly three blocks is admitted and forward publishes the
     * exhausted all-ones sentinel as its successor. */
    if (oi == 12u) {
      const int64_t expected_state[4] = {1, INT64_C(0x123456789abcdef), -1,
                                         -1};
      make_case(&c, &operations[oi]);
      ((int64_t *)c.inputs[2].data)[2] = -4;
      ((int64_t *)c.inputs[2].data)[3] = -1;
      success_inputs_unchanged(runtime, &c, "multi-block-max-successor");
      CHECK(memcmp(c.outputs[1].data, expected_state,
                   sizeof(expected_state)) == 0);
    }

    /* Both signed zero probabilities are valid byte-copy paths and do not
     * consume even the maximum counter sentinel. */
    for (size_t sign = 0; sign < 2u; sign++) {
      const uint32_t zero = sign ? UINT32_C(0x80000000) : 0u;
      make_case(&c, &operations[oi]);
      memcpy(c.inputs[1].data, &zero, sizeof(zero));
      ((int64_t *)c.inputs[2].data)[2] = -1;
      ((int64_t *)c.inputs[2].data)[3] = -1;
      success_inputs_unchanged(runtime, &c, "signed-zero-dropout");
      CHECK(memcmp(c.outputs[0].data, c.inputs[0].data,
                   c.outputs[0].byte_length) == 0);
      if (oi == 12u)
        CHECK(memcmp(c.outputs[1].data, c.inputs[2].data,
                     c.outputs[1].byte_length) == 0);
    }
  }
}

static void set_all_float_inputs(call_case *c, float value) {
  for (size_t i = 0; i < c->input_count; i++) {
    if (strcmp(c->inputs[i].dtype, "f32") != 0) continue;
    float *data = (float *)c->inputs[i].data;
    const size_t count = c->inputs[i].byte_length / sizeof(float);
    for (size_t j = 0; j < count; j++) data[j] = value;
  }
}

static void make_prospective_nonfinite(call_case *c,
                                       const operation_spec *spec) {
  make_case(c, spec);
  if (is_operation(spec, "embedding.backward")) {
    int64_t *ids = (int64_t *)c->inputs[0].data;
    for (size_t i = 0; i < 4u; i++) ids[i] = 0;
    {
      float *upstream = (float *)c->inputs[1].data;
      for (size_t i = 0; i < 8u; i++) upstream[i] = FLT_MAX;
    }
  } else if (begins(spec, "linear.")) {
    set_all_float_inputs(c, 2.0f);
    ((float *)c->inputs[0].data)[0] = FLT_MAX;
    if (begins(spec, "linear.backward"))
      ((float *)c->inputs[2].data)[0] = FLT_MAX;
  } else if (begins(spec, "layer-norm.")) {
    float *x = (float *)c->inputs[0].data;
    for (size_t i = 0; i < c->inputs[0].byte_length / sizeof(float); i++)
      x[i] = (i & 1u) ? -FLT_MAX : FLT_MAX;
  } else if (is_operation(spec, "gelu.backward")) {
    set_all_float_inputs(c, 1.0f);
    *(float *)c->inputs[1].data = FLT_MAX;
  } else if (is_operation(spec, "residual.forward")) {
    set_all_float_inputs(c, FLT_MAX);
  } else if (begins(spec, "dropout.train.")) {
    float *x = (float *)c->inputs[0].data;
    for (size_t i = 0; i < c->inputs[0].byte_length / sizeof(float); i++)
      x[i] = FLT_MAX;
    *(float *)c->inputs[1].data = 0.5f;
  }
}

static int prospective_nonfinite_operation(const operation_spec *spec) {
  return is_operation(spec, "embedding.backward") ||
         begins(spec, "linear.") || begins(spec, "layer-norm.") ||
         is_operation(spec, "gelu.backward") ||
         is_operation(spec, "residual.forward") ||
         begins(spec, "dropout.train.");
}

static unsigned mxcsr_status(void) {
#if defined(__x86_64__)
  return _mm_getcsr() & UINT32_C(0x3f);
#else
  return 0u;
#endif
}

static void raise_status(int flags) {
  CHECK(feclearexcept(FE_ALL_EXCEPT) == 0);
  CHECK(feraiseexcept(flags) == 0);
#if defined(__x86_64__)
  {
    const unsigned current = _mm_getcsr();
    const unsigned independent_status =
        ((unsigned)flags ^ UINT32_C(0x2a)) & UINT32_C(0x3f);
    _mm_setcsr((current & ~UINT32_C(0x3f)) | independent_status);
  }
#endif
  CHECK((fetestexcept(FE_ALL_EXCEPT) & flags) == flags);
}

static void validate_preserves_status(call_case *c, int expect_success,
                                      int flags, const char *label) {
  et_kernel_error error;
  int before;
  unsigned before_mxcsr;
  int32_t result;
  backing_snapshot snapshot;
  snapshot_backing(c, &snapshot);
  raise_status(flags);
  before = fetestexcept(FE_ALL_EXCEPT);
  before_mxcsr = mxcsr_status();
  result = et_n2_kernel_provider_v1()->validate_call(&c->call, &error);
  if ((expect_success && result != 0) || (!expect_success && result == 0)) {
    fprintf(stderr, "unexpected FP validation result %s/%s: %d %u/%u %s\n",
            c->request.operation, label, result, error.category, error.code,
            error.message);
    CHECK(0);
  }
  if (!expect_success) {
    CHECK(error.category == ET_KERNEL_ERROR_INVALID_ARGUMENT);
    CHECK(error.code == ET_KERNEL_CODE_PROVIDER_REJECTED);
  }
  CHECK(fetestexcept(FE_ALL_EXCEPT) == before);
  if (mxcsr_status() != before_mxcsr) {
    fprintf(stderr,
            "MXCSR status changed for %s/%s flags=%x: before=%x after=%x\n",
            c->request.operation, label, flags, before_mxcsr, mxcsr_status());
    CHECK(0);
  }
  check_backing(c, &snapshot);
  CHECK(feclearexcept(FE_ALL_EXCEPT) == 0);
}

static void test_fp_status(et_kernel_runtime *runtime) {
  static const int status_sets[] = {
      FE_INVALID, FE_DIVBYZERO, FE_OVERFLOW, FE_UNDERFLOW, FE_INEXACT,
      FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW | FE_INEXACT};
  (void)runtime;
  for (size_t oi = 0; oi < COUNT(operations); oi++) {
    for (size_t status = 0; status < COUNT(status_sets); status++) {
      call_case c;
      make_case(&c, &operations[oi]);
      validate_preserves_status(&c, 1, status_sets[status], "success");
      make_case(&c, &operations[oi]);
      for (size_t input = 0; input < c.input_count; input++) {
        if (strcmp(c.inputs[input].dtype, "f32") == 0) {
          const uint32_t nan_bits = UINT32_C(0x7fc00001);
          memcpy(c.inputs[input].data, &nan_bits, sizeof(nan_bits));
          break;
        }
      }
      validate_preserves_status(&c, 0, status_sets[status],
                                "nonfinite-rejection");
      if (prospective_nonfinite_operation(&operations[oi])) {
        make_prospective_nonfinite(&c, &operations[oi]);
        validate_preserves_status(&c, 0, status_sets[status],
                                  "preflight-nonfinite");
      }
    }
  }
}

static void test_prospective_nonfinite(et_kernel_runtime *runtime) {
  for (size_t oi = 0; oi < COUNT(operations); oi++) {
    call_case c;
    if (!prospective_nonfinite_operation(&operations[oi])) continue;
    make_prospective_nonfinite(&c, &operations[oi]);
    reject_unchanged(runtime, &c, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_PROVIDER_REJECTED,
                     "finite-input-nonfinite-result");
  }
}

int main(void) {
  et_kernel_runtime *runtime = new_runtime();
  for (size_t operation = 0; operation < COUNT(operations); operation++) {
    test_schema_and_views(runtime, &operations[operation]);
    test_aliases(runtime, &operations[operation]);
    test_nonfinite_inputs(runtime, &operations[operation]);
  }
  test_embedding_ids(runtime);
  test_epsilon(runtime);
  test_dropout_state(runtime);
  test_prospective_nonfinite(runtime);
  test_fp_status(runtime);
  et_kernel_runtime_destroy(runtime);
  CHECK(feclearexcept(FE_ALL_EXCEPT) == 0);
  printf("N2 negative atomicity PASS (%zu checks)\n", checks);
  return 0;
}
