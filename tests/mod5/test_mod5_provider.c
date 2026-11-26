#include "mod5_test.h"

#include <errno.h>
#include <fenv.h>
#include <float.h>
#include <xmmintrin.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

enum { ELEMENTS = 1024, MAX_POSITIONS = 16, MAX_MASK = 256 };

static const uint64_t rms_rows[][3] = {
    {1, 1, 1}, {1, 1, 4}, {1, 2, 4}, {2, 3, 4}, {2, 3, 8},
    {1, 8, 16}, {2, 8, 16}, {1, 16, 32}};
static const uint64_t swi_rows[][3] = {
    {1, 1, 1}, {1, 1, 8}, {1, 2, 4}, {2, 3, 4}, {2, 3, 8},
    {1, 8, 16}, {1, 8, 32}, {2, 8, 32}, {1, 16, 64}};
static const uint64_t attention_rows[][6] = {
    {1, 2, 1, 2, 2, 2}, {1, 4, 1, 3, 3, 2}, {2, 4, 1, 2, 3, 4},
    {1, 4, 2, 3, 3, 2}, {2, 4, 2, 2, 3, 4}, {1, 2, 2, 3, 3, 2},
    {1, 4, 1, 8, 8, 4}, {1, 4, 2, 8, 8, 4}, {2, 4, 2, 8, 8, 4},
    {1, 4, 1, 16, 16, 8}, {1, 4, 2, 16, 16, 8}};

typedef struct fixture {
  et_kernel_request_v1 forward_request, backward_request;
  et_kernel_call_v1 forward, backward;
  et_kernel_tensor_view_v1 inputs[9], output, grads[3];
  uint64_t q[4], kv[4], qp[2], kp[2], mask_shape[3];
  float operands[3][ELEMENTS], upstream[ELEMENTS], slopes[4], epsilon;
  float result[ELEMENTS], gradients[3][ELEMENTS];
  int64_t query_pos[MAX_POSITIONS], key_pos[MAX_POSITIONS], window;
  uint8_t mask[MAX_MASK];
} fixture;

static void set_views(fixture *f, const char *cap, const char *forward,
                      const char *backward, const uint64_t *shape, size_t rank,
                      size_t forward_inputs, size_t gradients) {
  f->forward_request = (et_kernel_request_v1){sizeof(et_kernel_request_v1),
                                              forward,
                                              "f32",
                                              "cpu",
                                              rank,
                                              shape,
                                              1u,
                                              {0}};
  f->backward_request = f->forward_request;
  f->backward_request.operation = backward;
  f->forward = test_call(cap, &f->forward_request, forward_inputs, f->inputs,
                         1u, &f->output);
  f->backward = test_call(cap, &f->backward_request, forward_inputs + 1u,
                          f->inputs, gradients, f->grads);
  f->inputs[forward_inputs] =
      test_view(f->upstream, "f32", f->output.rank, f->output.shape);
}

static void elementwise_fixture(fixture *f, const uint64_t *s, int swi) {
  memset(f, 0, sizeof(*f));
  f->epsilon = 0.0625f;
  f->inputs[0] = test_view(f->operands[0], "f32", 3u, s);
  f->inputs[1] = swi ? test_view(f->operands[1], "f32", 3u, s)
                     : test_view(f->operands[1], "f32", 1u, s + 2u);
  f->inputs[2] = test_view(&f->epsilon, "f32", 0u, NULL);
  f->output = test_view(f->result, "f32", 3u, s);
  for (size_t i = 0; i < 2u; ++i) {
    f->grads[i] = f->inputs[i];
    f->grads[i].data = f->gradients[i];
  }
  if (swi)
    set_views(f, "kernel.swiglu", "swiglu.forward", "swiglu.backward", s, 3u,
              2u, 2u);
  else
    set_views(f, "kernel.rmsnorm", "rmsnorm.forward", "rmsnorm.backward", s, 3u,
              3u, 2u);
}

static void attention_fixture(fixture *f, const uint64_t *s, int mode) {
  memset(f, 0, sizeof(*f));
  const uint64_t q[] = {s[0], s[1], s[3], s[5]};
  const uint64_t kv[] = {s[0], s[2], s[4], s[5]};
  const uint64_t qp[] = {s[0], s[3]};
  const uint64_t kp[] = {s[0], s[4]};
  const uint64_t mask[] = {s[0], s[3], s[4]};
  memcpy(f->q, q, sizeof(q));
  memcpy(f->kv, kv, sizeof(kv));
  memcpy(f->qp, qp, sizeof(qp));
  memcpy(f->kp, kp, sizeof(kp));
  memcpy(f->mask_shape, mask, sizeof(mask));
  /* Mode 0 has selective masking and ALiBi, 1 leaves zeros and a
     one-position window, 2 masks all, 3 uses the largest exact delta,
     and 4 disables ALiBi with a wide window. */
  f->window = mode == 1 ? 1 : mode >= 3 ? INT64_MAX : 8;
  for (size_t n = 0; n < s[0]; ++n) {
    for (size_t t = 0; t < s[3]; ++t) {
      f->query_pos[n * s[3] + t] =
          mode == 3 ? ET_MOD5_MAX_EXACT_POSITION - (int64_t)(s[3] - 1u - t)
                    : (int64_t)(t * 2u + n);
      for (size_t k = 0; k < s[4]; ++k) {
        f->mask[(n * s[3] + t) * s[4] + k] =
            mode != 2 && (mode != 0 || (n + t + k) % 5u != 2u);
      }
    }
    for (size_t k = 0; k < s[4]; ++k) {
      f->key_pos[n * s[4] + k] =
          mode == 3 ? (int64_t)k : (int64_t)(k * 2u + 2u * n);
    }
  }
  if (mode != 1 && mode != 4) {
    for (size_t h = 0; h < s[1]; ++h)
      f->slopes[h] = ((float)h - 1.25f) * (mode == 3 ? 0x1p-27f : 0.09375f);
  }
  f->inputs[0] = test_view(f->operands[0], "f32", 4u, f->q);
  f->inputs[1] = test_view(f->operands[1], "f32", 4u, f->kv);
  f->inputs[2] = test_view(f->operands[2], "f32", 4u, f->kv);
  f->inputs[3] = test_view(f->query_pos, "i64", 2u, f->qp);
  f->inputs[4] = test_view(f->key_pos, "i64", 2u, f->kp);
  f->inputs[5] = test_view(f->mask, "bool", 3u, f->mask_shape);
  f->inputs[6] = test_view(f->slopes, "f32", 1u, s + 1u);
  f->inputs[7] = test_view(&f->window, "i64", 0u, NULL);
  f->output = test_view(f->result, "f32", 4u, f->q);
  for (size_t i = 0; i < 3u; ++i) {
    f->grads[i] = f->inputs[i];
    f->grads[i].data = f->gradients[i];
  }
  set_views(f, "kernel.variant-attention", "variant-attention.forward",
            "variant-attention.backward", s, 6u, 8u, 3u);
}

static void seed_values(fixture *f) {
  for (size_t field = 0; field < f->backward.output_count; ++field) {
    for (size_t i = 0; i < f->grads[field].byte_length / sizeof(float); ++i) {
      const size_t value = (i * (7u + 4u * field) + 5u * field + i / 3u) % 31u;
      f->operands[field][i] = ((float)value - 15.0f) * 0.0625f;
    }
  }
  for (size_t i = 0; i < f->output.byte_length / sizeof(float); ++i)
    f->upstream[i] = ((float)((i * 13u + i / 5u + 3u) % 23u) - 11.0f) * 0.0625f;
}

static double objective(const et_kernel_runtime *runtime, fixture *f) {
  dispatch(runtime, &f->forward);
  double sum = 0.0;
  for (size_t i = 0; i < f->output.byte_length / sizeof(float); ++i)
    sum += (double)f->upstream[i] * (double)f->result[i];
  return sum;
}

static size_t check_repeat_and_gradients(const et_kernel_runtime *runtime,
                                         fixture *f, int fully_masked) {
  float saved_result[ELEMENTS], saved_grads[3][ELEMENTS];
  dispatch(runtime, &f->forward);
  dispatch(runtime, &f->backward);
  memcpy(saved_result, f->result, sizeof(saved_result));
  memcpy(saved_grads, f->gradients, sizeof(saved_grads));
  memset(f->result, 0xa5, f->output.byte_length);
  for (size_t field = 0; field < f->backward.output_count; ++field)
    memset(f->gradients[field], 0xa5, f->grads[field].byte_length);
  dispatch(runtime, &f->forward);
  dispatch(runtime, &f->backward);
  CHECK(memcmp(saved_result, f->result, sizeof(saved_result)) == 0);
  CHECK(memcmp(saved_grads, f->gradients, sizeof(saved_grads)) == 0);
  if (fully_masked) {
    static const float zero[ELEMENTS] = {0.0f};
    CHECK(memcmp(zero, f->result, f->output.byte_length) == 0);
    for (size_t field = 0; field < f->backward.output_count; ++field)
      CHECK(memcmp(zero, f->gradients[field], f->grads[field].byte_length) ==
            0);
  }
  const float step = 0x1p-10f;
  const int wide_attention = f->forward_request.rank == 6u && f->q[2] >= 8u;
  size_t checked = 0;
  for (size_t field = 0; field < f->backward.output_count; ++field) {
    const size_t elements = f->grads[field].byte_length / sizeof(float);
    const size_t samples = wide_attention && elements > 24u ? 24u : elements;
    for (size_t sample = 0; sample < samples; ++sample) {
      const size_t i = samples == 1u ? 0u : sample * (elements - 1u) / (samples - 1u);
      const float original = f->operands[field][i];
      f->operands[field][i] = original + step;
      const double plus = objective(runtime, f);
      f->operands[field][i] = original - step;
      const double minus = objective(runtime, f);
      f->operands[field][i] = original;
      check_close(f->gradients[field][i], (plus - minus) / (2.0 * step),
                  0.00045, 0.002);
      ++checked;
    }
  }
  return checked;
}

static void dump_view(const et_kernel_tensor_view_v1 *view) {
  if (strcmp(view->dtype, "f32") == 0) {
    const float *values = view->data;
    for (size_t i = 0; i < view->byte_length / sizeof(*values); ++i)
      printf("%s%a", i == 0u ? "" : " ", (double)values[i]);
  } else if (strcmp(view->dtype, "i64") == 0) {
    const int64_t *values = view->data;
    for (size_t i = 0; i < view->byte_length / sizeof(*values); ++i)
      printf("%s%lld", i == 0u ? "" : " ", (long long)values[i]);
  } else {
    const uint8_t *values = view->data;
    for (size_t i = 0; i < view->byte_length; ++i)
      printf("%s%u", i == 0u ? "" : " ", (unsigned)values[i]);
  }
  putchar('\n');
}

static void dump_fixture(const et_kernel_runtime *runtime, fixture *f) {
  dispatch(runtime, &f->forward);
  dispatch(runtime, &f->backward);
  printf("%s", f->forward_request.operation);
  for (size_t d = 0; d < f->forward_request.rank; ++d)
    printf(" %llu", (unsigned long long)f->forward_request.shape[d]);
  putchar('\n');
  for (size_t i = 0; i < f->backward.input_count; ++i)
    dump_view(&f->inputs[i]);
  dump_view(&f->output);
  for (size_t i = 0; i < f->backward.output_count; ++i)
    dump_view(&f->grads[i]);
}

static void check_cases(const et_kernel_runtime *runtime, int oracle) {
  fixture f;
  size_t checked = 0;
  for (size_t i = 0; i < ARRAY_COUNT(rms_rows); ++i) {
    elementwise_fixture(&f, rms_rows[i], 0);
    seed_values(&f);
    if (oracle)
      dump_fixture(runtime, &f);
    else
      checked += check_repeat_and_gradients(runtime, &f, 0);
  }
  for (size_t i = 0; i < ARRAY_COUNT(swi_rows); ++i) {
    elementwise_fixture(&f, swi_rows[i], 1);
    seed_values(&f);
    if (oracle)
      dump_fixture(runtime, &f);
    else
      checked += check_repeat_and_gradients(runtime, &f, 0);
  }
  for (size_t i = 0; i < ARRAY_COUNT(attention_rows); ++i) {
    for (int mode = 0; mode < 5; ++mode) {
      attention_fixture(&f, attention_rows[i], mode);
      seed_values(&f);
      if (oracle)
        dump_fixture(runtime, &f);
      else
        checked += check_repeat_and_gradients(runtime, &f, mode == 2);
    }
  }
  if (!oracle)
    printf("MOD5 primitive finite differences: %zu checks\n", checked);
}

static void check_capability(const et_kernel_runtime *runtime, const char *name,
                             const char *forward, const char *backward,
                             size_t rank, size_t count,
                             const uint64_t rows[count][rank]) {
  const et_kernel_capability_v1 *cap =
      et_kernel_runtime_capability_find(runtime, name);
  CHECK(cap && cap->status == ET_KERNEL_CAPABILITY_VERIFIED &&
        cap->deterministic == 1u);
  CHECK(strcmp(cap->implementation, "mod5.cpu-f32.serial") == 0);
  CHECK(strcmp(cap->version, "1.0") == 0);
  CHECK(strcmp(cap->evidence, "MOD5:cpu-f32-variants-v1") == 0);
  CHECK(cap->operation_count == 2u && cap->dtype_count == 1u &&
        cap->device_count == 1u);
  CHECK(strcmp(cap->dtypes[0], "f32") == 0 &&
        strcmp(cap->devices[0], "cpu") == 0);
  CHECK(cap->shape_range_count == count);
  for (size_t r = 0; r < count; ++r) {
    et_kernel_request_v1 request = {
        sizeof(request), forward, "f32", "cpu", rank, rows[r], 1u, {0}};
    et_kernel_error error;
    CHECK(et_kernel_runtime_capability_require(runtime, name, &request, NULL,
                                               &error) == 0);
    request.operation = backward;
    CHECK(et_kernel_runtime_capability_require(runtime, name, &request, NULL,
                                               &error) == 0);
    size_t matches = 0u;
    for (size_t candidate = 0; candidate < count; ++candidate) {
      const et_kernel_shape_range_v1 *range = &cap->shape_ranges[candidate];
      CHECK(range->rank == rank);
      size_t d = 0u;
      for (; d < rank; ++d) {
        CHECK(range->dimensions[d].minimum == range->dimensions[d].maximum);
        CHECK(range->dimensions[d].maximum_unbounded == 0u);
        if (range->dimensions[d].minimum != rows[r][d])
          break;
      }
      if (d == rank)
        ++matches;
    }
    CHECK(matches == 1u);
  }
}

static void check_metadata(const et_kernel_runtime *runtime) {
  const et_kernel_provider_v1 *provider = et_mod5_kernel_provider_v1();
  CHECK(provider && provider->struct_size == ET_KERNEL_PROVIDER_V1_0_SIZE);
  CHECK(provider->abi_major == 1u && provider->abi_minor == 0u);
  CHECK(provider->required_features == 0u && provider->capability_count == 3u);
  CHECK(provider->capability_stride == ET_KERNEL_CAPABILITY_V1_0_SIZE);
  CHECK(provider->capability_bytes == 3u * provider->capability_stride);
  CHECK(strcmp(provider->name, "mod5.cpu-f32.serial") == 0);
  check_capability(runtime, "kernel.rmsnorm", "rmsnorm.forward",
                   "rmsnorm.backward", 3u, ARRAY_COUNT(rms_rows), rms_rows);
  check_capability(runtime, "kernel.swiglu", "swiglu.forward",
                   "swiglu.backward", 3u, ARRAY_COUNT(swi_rows), swi_rows);
  check_capability(runtime, "kernel.variant-attention",
                   "variant-attention.forward", "variant-attention.backward",
                   6u, ARRAY_COUNT(attention_rows), attention_rows);

  et_kernel_runtime *baseline = NULL;
  et_kernel_error error;
  fixture f;
  attention_fixture(&f, attention_rows[0], 0);
  CHECK(et_kernel_runtime_baseline(&baseline, &error) == 0);
  CHECK(et_kernel_runtime_capability_require(
            baseline, f.forward.capability, &f.forward_request, NULL, &error) ==
        ET_KERNEL_ERROR_UNSUPPORTED);
  CHECK(error.code == ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
  et_kernel_runtime_destroy(baseline);

  char report[16384];
  size_t capacity;
  CHECK(et_kernel_runtime_report_json(runtime, report, sizeof(report),
                                      &capacity, &error) == 0);
  CHECK(strstr(report, "kernel.rmsnorm") && strstr(report, "kernel.swiglu") &&
        strstr(report, "kernel.variant-attention"));
}

static void check_rejected(const et_kernel_runtime *runtime, fixture *f,
                           const et_kernel_call_v1 *call,
                           et_kernel_error_category category,
                           et_kernel_error_code k1_code,
                           et_kernel_error_code direct_code) {
  memset(f->result, 0xa5, sizeof(f->result));
  memset(f->gradients, 0x5a, sizeof(f->gradients));
  const fixture before = *f;
  et_kernel_error error;
  int32_t status = et_kernel_runtime_dispatch(runtime, call, &error);
  if (status != (int32_t)category || error.category != category ||
      error.code != k1_code)
    fprintf(stderr, "K1 observed %d/%u %s, expected %u/%u\n", status,
            error.code, error.message, category, k1_code);
  CHECK(status == (int32_t)category && error.category == category &&
        error.code == k1_code);
  CHECK(memcmp(before.result, f->result, sizeof(f->result)) == 0);
  CHECK(memcmp(before.gradients, f->gradients, sizeof(f->gradients)) == 0);
  CHECK(memcmp(before.operands, f->operands, sizeof(f->operands)) == 0);
  CHECK(memcmp(before.upstream, f->upstream, sizeof(f->upstream)) == 0);
  CHECK(memcmp(before.slopes, f->slopes, sizeof(f->slopes)) == 0);
  CHECK(memcmp(&before.epsilon, &f->epsilon, sizeof(f->epsilon)) == 0);
  CHECK(memcmp(before.query_pos, f->query_pos, sizeof(f->query_pos)) == 0);
  CHECK(memcmp(before.key_pos, f->key_pos, sizeof(f->key_pos)) == 0);
  CHECK(memcmp(before.mask, f->mask, sizeof(f->mask)) == 0 &&
        f->window == before.window);
  status = et_mod5_kernel_provider_v1()->validate_call(call, &error);
  if (status != (int32_t)category || error.category != category ||
      error.code != direct_code)
    fprintf(stderr, "direct observed %d/%u %s, expected %u/%u\n", status,
            error.code, error.message, category, direct_code);
  CHECK(status == (int32_t)category && error.category == category &&
        error.code == direct_code);
  CHECK(memcmp(&before.operands, &f->operands, sizeof(f->operands)) == 0);
  CHECK(memcmp(before.result, f->result, sizeof(f->result)) == 0);
  CHECK(memcmp(before.gradients, f->gradients, sizeof(f->gradients)) == 0);
}

#define REJECT(call, category, code)                                           \
  check_rejected(runtime, &f, call, ET_KERNEL_ERROR_##category,                \
                 ET_KERNEL_CODE_##code, ET_KERNEL_CODE_##code)
#define NUMERIC(call) REJECT(call, INVALID_ARGUMENT, PROVIDER_REJECTED)
#define UNSUPPORTED(call)                                                      \
  check_rejected(runtime, &f, call, ET_KERNEL_ERROR_UNSUPPORTED,               \
                 ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED,                       \
                 ET_KERNEL_CODE_PROVIDER_REJECTED)

static void check_malformed(const et_kernel_runtime *runtime) {
  fixture f;
  elementwise_fixture(&f, rms_rows[2], 0);
  seed_values(&f);
  const fixture good = f;
  f.forward.struct_size = 8u;
  REJECT(&f.forward, VERSION_MISMATCH, INVALID_STRUCT_SIZE);
  f = good;
  f.forward_request.struct_size = 8u;
  REJECT(&f.forward, VERSION_MISMATCH, INVALID_STRUCT_SIZE);
  f = good;
  f.inputs[1].struct_size = 8u;
  REJECT(&f.backward, VERSION_MISMATCH, INVALID_STRUCT_SIZE);
  f = good;
  f.forward.input_stride = 8u;
  REJECT(&f.forward, VERSION_MISMATCH, INVALID_STRUCT_SIZE);
  f = good;
  --f.forward.input_bytes;
  REJECT(&f.forward, INVALID_ARGUMENT, INTEGER_OVERFLOW);
  f = good;
  f.forward.inputs = (const unsigned char *)f.inputs + 1u;
  REJECT(&f.forward, INVALID_ARGUMENT, INVALID_BUFFER);
  f = good;
  f.backward.output_count = 1u;
  f.backward.output_bytes = sizeof(f.grads[0]);
  REJECT(&f.backward, INVALID_ARGUMENT, INVALID_BUFFER);
  f = good;
  f.inputs[0].offset_bytes = 1u;
  REJECT(&f.backward, NONCONTIGUOUS, INVALID_BUFFER);
  f = good;
  f.inputs[0].device = "gpu";
  REJECT(&f.forward, DEVICE_MISMATCH, INVALID_BUFFER);
  f = good;
  f.inputs[0].dtype = "bad";
  REJECT(&f.forward, DTYPE_MISMATCH, INVALID_TEXT);
  f = good;
  f.inputs[1] = test_view(f.operands[1], "f32", 1u, (const uint64_t[]){2u});
  REJECT(&f.backward, SHAPE_MISMATCH, INVALID_SHAPE);
  f = good;
  f.inputs[0].shape =
      (const uint64_t *)((const unsigned char *)rms_rows[2] + 1u);
  REJECT(&f.forward, INVALID_ARGUMENT, INVALID_BUFFER);
  f = good;
  --f.grads[1].byte_length;
  REJECT(&f.backward, SHAPE_MISMATCH, INVALID_BUFFER);
  f = good;
  f.inputs[1].data = (unsigned char *)f.operands[1] + 1u;
  REJECT(&f.forward, INVALID_ARGUMENT, PROVIDER_REJECTED);
  f = good;
  f.inputs[1].data = (void *)(UINTPTR_MAX & ~(uintptr_t)3u);
  check_rejected(runtime, &f, &f.forward, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                 ET_KERNEL_CODE_ALIASING_OUTPUT,
                 ET_KERNEL_CODE_PROVIDER_REJECTED);
  f = good;
  f.inputs[1].data = f.operands[0] + 2u;
  REJECT(&f.backward, INVALID_ARGUMENT, PROVIDER_REJECTED);
  f = good;
  f.grads[1].data = f.operands[0] + 2u;
  REJECT(&f.backward, INVALID_ARGUMENT, ALIASING_OUTPUT);
  f = good;
  f.grads[1].data = f.gradients[0] + 4u;
  REJECT(&f.backward, INVALID_ARGUMENT, ALIASING_OUTPUT);
  f = good;
  f.forward_request.operation = "rmsnorm.unknown";
  UNSUPPORTED(&f.forward);
  f = good;
  f.forward_request.shape = (const uint64_t[]){1, 2, 2};
  UNSUPPORTED(&f.forward);
  f = good;
  f.forward_request.shape = (const uint64_t[]){0, 2, 4};
  UNSUPPORTED(&f.forward);
  f = good;
  f.forward_request.dtype = "f16";
  UNSUPPORTED(&f.forward);
  f = good;
  f.forward_request.device = "gpu";
  for (size_t i = 0; i < f.forward.input_count; ++i)
    f.inputs[i].device = "gpu";
  f.output.device = "gpu";
  UNSUPPORTED(&f.forward);

  attention_fixture(&f, attention_rows[3], 0);
  f.forward_request.shape = (const uint64_t[]){1, 4, 3, 3, 3, 2};
  UNSUPPORTED(&f.forward);
  f.forward_request.shape = (const uint64_t[]){1, 2, 4, 3, 3, 2};
  UNSUPPORTED(&f.forward);
  f.forward_request.shape = (const uint64_t[]){1, 4, 0, 3, 3, 2};
  UNSUPPORTED(&f.forward);
  f.forward_request.shape = (const uint64_t[]){1, 4, 2, 2, 3, 2};
  UNSUPPORTED(&f.forward);
}

static void check_numeric_errors(const et_kernel_runtime *runtime) {
  fixture f;
  elementwise_fixture(&f, rms_rows[4], 0);
  seed_values(&f);
  const float bad_epsilon[] = {0.0f, -0.0f, -1.0f, INFINITY, NAN};
  for (size_t i = 0; i < ARRAY_COUNT(bad_epsilon); ++i) {
    f.epsilon = bad_epsilon[i];
    NUMERIC(&f.forward);
    NUMERIC(&f.backward);
  }
  f.epsilon = 0.0625f;
  f.operands[0][47] = FLT_MAX;
  NUMERIC(&f.forward);
  f.operands[0][47] = 1.0f;
  f.upstream[47] = FLT_MAX;
  f.operands[1][7] = 2.0f;
  NUMERIC(&f.backward);
  f.upstream[47] = 0.5f;
  f.operands[1][7] = NAN;
  NUMERIC(&f.backward);
  for (size_t i = 0; i < 48u; ++i)
    f.operands[0][i] = 1.0f;
  f.operands[1][7] = 0.0f;
  f.upstream[39] = f.upstream[47] = 0.75f * FLT_MAX;
  NUMERIC(&f.backward); /* dGamma accumulation overflows with finite dX. */

  elementwise_fixture(&f, swi_rows[4], 1);
  seed_values(&f);
  f.operands[0][47] = -100.0f;
  NUMERIC(&f.forward);
  NUMERIC(&f.backward);
  f.operands[0][47] = 8.0f;
  f.operands[1][47] = FLT_MAX;
  NUMERIC(&f.forward);
  f.upstream[47] = 2.0f;
  NUMERIC(&f.backward);
  f.operands[1][47] = -INFINITY;
  NUMERIC(&f.forward);

  attention_fixture(&f, attention_rows[4], 3);
  seed_values(&f);
  f.window = 0;
  NUMERIC(&f.forward);
  f.window = INT64_MIN;
  NUMERIC(&f.backward);
  f.window = INT64_MAX;
  f.mask[11] = 2u;
  NUMERIC(&f.backward);
  f.mask[11] = 1u;
  f.query_pos[3] = ET_MOD5_MAX_EXACT_POSITION + 1;
  NUMERIC(&f.forward);
  f.query_pos[3] = f.query_pos[2];
  NUMERIC(&f.backward);
  f.query_pos[3] = ET_MOD5_MAX_EXACT_POSITION;
  f.key_pos[3] = -1;
  NUMERIC(&f.forward);
  f.key_pos[3] = 0;
  f.key_pos[5] = f.key_pos[4];
  NUMERIC(&f.backward);
  f.key_pos[5] = 2;
  f.slopes[3] = INFINITY;
  NUMERIC(&f.forward);
  f.slopes[3] = FLT_MAX;
  NUMERIC(&f.forward); /* Overflow in the ALiBi product for the last head. */
  f.slopes[3] = 0.0f;
  f.operands[0][63] = FLT_MAX;
  f.operands[1][47] = 2.0f;
  NUMERIC(&f.forward); /* Last row's dot product overflows. */
  f.operands[0][63] = 0.5f;
  f.upstream[63] = FLT_MAX;
  f.operands[2][47] = -2.0f;
  NUMERIC(&f.backward);
  f.upstream[63] = -0.5f;
  f.operands[2][47] = NAN;
  f.mask[11] = 0u;
  NUMERIC(&f.forward); /* Nonfinite inputs reject even when masked. */

  attention_fixture(&f, attention_rows[0], 4);
  f.operands[0][2] = 1.0f;
  f.operands[1][0] = FLT_MAX;
  f.operands[1][2] = -FLT_MAX;
  NUMERIC(&f.forward); /* Both scores are finite, but score - max overflows. */
  NUMERIC(&f.backward);

  attention_fixture(&f, attention_rows[0], 4);
  f.mask[3] = 0u;
  f.upstream[0] = f.upstream[6] = 0.75f * FLT_MAX;
  NUMERIC(&f.backward); /* Two query heads overflow one shared dV element. */
}

static void check_wide_numeric_errors(const et_kernel_runtime *runtime) {
  fixture f;
  for (size_t row = 5u; row < ARRAY_COUNT(rms_rows); ++row) {
    elementwise_fixture(&f, rms_rows[row], 0);
    const size_t last = f.output.byte_length / sizeof(float) - 1u;
    seed_values(&f);
    f.operands[0][last] = NAN;
    NUMERIC(&f.forward);
    NUMERIC(&f.backward);
    f.operands[0][last] = FLT_MAX;
    NUMERIC(&f.forward);
    NUMERIC(&f.backward);
    f.operands[0][last] = 1.0f;
    f.epsilon = -1.0f;
    NUMERIC(&f.forward);
    NUMERIC(&f.backward);
    f.epsilon = 0.0625f;

    /* Make the final two rows overflow one dGamma element with finite dX. */
    const size_t width = rms_rows[row][2];
    memset(f.upstream, 0, sizeof(f.upstream));
    for (size_t i = 0; i <= last; ++i)
      f.operands[0][i] = 1.0f;
    f.operands[1][width - 1u] = 0.0f;
    f.upstream[last - width] = f.upstream[last] = 0.75f * FLT_MAX;
    NUMERIC(&f.backward);
  }

  for (size_t row = 5u; row < ARRAY_COUNT(swi_rows); ++row) {
    elementwise_fixture(&f, swi_rows[row], 1);
    const size_t last = f.output.byte_length / sizeof(float) - 1u;
    seed_values(&f);
    f.operands[0][last] = INFINITY;
    NUMERIC(&f.forward);
    NUMERIC(&f.backward);
    f.operands[0][last] = -100.0f;
    NUMERIC(&f.forward);
    NUMERIC(&f.backward);
    f.operands[0][last] = 8.0f;
    f.operands[1][last] = FLT_MAX;
    f.upstream[last] = 2.0f;
    NUMERIC(&f.forward);
    NUMERIC(&f.backward);
  }

  for (size_t row = 6u; row < ARRAY_COUNT(attention_rows); ++row) {
    const uint64_t *shape = attention_rows[row];
    attention_fixture(&f, shape, 4);
    const size_t last = f.output.byte_length / sizeof(float) - 1u;
    seed_values(&f);
    f.operands[0][last] = NAN;
    NUMERIC(&f.forward);
    NUMERIC(&f.backward);
    f.operands[0][last] = 1.0f;
    f.window = -1;
    NUMERIC(&f.forward);
    NUMERIC(&f.backward);
    f.window = INT64_MAX;
    f.mask[f.inputs[5].byte_length - 1u] = 2u;
    NUMERIC(&f.forward);
    NUMERIC(&f.backward);
    f.mask[f.inputs[5].byte_length - 1u] = 1u;
    f.operands[0][last] = FLT_MAX;
    const size_t last_group = (shape[0] * shape[2] - 1u) * shape[4] * shape[5];
    f.operands[1][last_group + shape[5] - 1u] = 2.0f;
    NUMERIC(&f.forward);
    NUMERIC(&f.backward);

    /* One key at token zero gets two last-group, same-row contributions. */
    attention_fixture(&f, shape, 4);
    const size_t query_row = shape[3] * shape[5];
    const size_t q_batch = (shape[0] - 1u) * shape[1];
    const size_t q_token = shape[3] - 1u;
    for (size_t k = 1; k < shape[4]; ++k)
      f.mask[((shape[0] - 1u) * shape[3] + q_token) * shape[4] + k] = 0;
    const size_t index = (q_batch + shape[1] - 1u) * query_row + q_token * shape[5];
    f.upstream[index] = f.upstream[index - query_row] = 0.75f * FLT_MAX;
    NUMERIC(&f.backward);
  }
}

static uint16_t read_x87(void) {
  uint16_t word;
  __asm__ volatile("fnstcw %0" : "=m"(word));
  return word;
}

static void write_x87(uint16_t word) {
  __asm__ volatile("fldcw %0" : : "m"(word));
}

static void check_fenv(const et_kernel_runtime *runtime) {
  fixture f;
  elementwise_fixture(&f, swi_rows[0], 1);
  seed_values(&f);
  fenv_t saved;
  CHECK(fegetenv(&saved) == 0);
  CHECK(feclearexcept(FE_ALL_EXCEPT) == 0);
  const unsigned baseline = _mm_getcsr();
  const uint16_t x87 = read_x87();
  const unsigned unsupported[] = {baseline | 0x8000u, baseline | 0x0040u,
                                  baseline | 0x2000u, baseline & ~0x0080u};
  for (size_t i = 0; i < ARRAY_COUNT(unsupported); ++i) {
    _mm_setcsr(unsupported[i]);
    REJECT(&f.forward, UNSUPPORTED, PROVIDER_REJECTED);
    CHECK(_mm_getcsr() == unsupported[i] && read_x87() == x87);
  }
  _mm_setcsr(baseline);
  const uint16_t x87_modes[] = {(uint16_t)(x87 | 0x0400u),
                                (uint16_t)(x87 & ~0x0001u)};
  for (size_t i = 0; i < ARRAY_COUNT(x87_modes); ++i) {
    write_x87(x87_modes[i]);
    REJECT(&f.forward, UNSUPPORTED, PROVIDER_REJECTED);
    CHECK(read_x87() == x87_modes[i] && _mm_getcsr() == baseline);
  }
  write_x87(x87);
  CHECK(fesetround(FE_DOWNWARD) == 0);
  REJECT(&f.forward, UNSUPPORTED, PROVIDER_REJECTED);
  CHECK(fegetround() == FE_DOWNWARD);
  CHECK(fesetround(FE_TONEAREST) == 0);

  et_kernel_error error;
  const et_kernel_provider_v1 *provider = et_mod5_kernel_provider_v1();
  CHECK(feraiseexcept(FE_DIVBYZERO) == 0);
  const int sticky = fetestexcept(FE_ALL_EXCEPT);
  const unsigned before = _mm_getcsr();
  const float sentinel = f.result[0];
  errno = EDOM;
  f.operands[0][0] =
      104.0f; /* expf underflow is finite; preflight preserves errno. */
  CHECK(provider->validate_call(&f.forward, &error) == 0);
  CHECK(f.result[0] == sentinel && fetestexcept(FE_ALL_EXCEPT) == sticky &&
        _mm_getcsr() == before && read_x87() == x87 && errno == EDOM);
  f.operands[0][0] =
      -104.0f; /* expf overflow rejects and preserves caller state. */
  CHECK(provider->validate_call(&f.forward, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(f.result[0] == sentinel && fetestexcept(FE_ALL_EXCEPT) == sticky &&
        _mm_getcsr() == before && read_x87() == x87 && errno == EDOM);
  CHECK(fesetenv(&saved) == 0);
}

static void check_strided_tables(const et_kernel_runtime *runtime) {
  fixture f;
  elementwise_fixture(&f, rms_rows[2], 0);
  seed_values(&f);
  dispatch(runtime, &f.backward);
  float expected[3][ELEMENTS];
  memcpy(expected, f.gradients, sizeof(expected));

  struct extended_view {
    et_kernel_tensor_view_v1 view;
    uint64_t extension;
  } in[4], out[2];
  for (size_t i = 0; i < ARRAY_COUNT(in); ++i)
    in[i] = (struct extended_view){f.inputs[i], UINT64_MAX};
  for (size_t i = 0; i < ARRAY_COUNT(out); ++i)
    out[i] = (struct extended_view){f.grads[i], UINT64_MAX};
  in[0].view.struct_size = sizeof(in[0]);
  out[0].view.struct_size = sizeof(out[0]);
  f.backward.inputs = in;
  f.backward.input_stride = sizeof(in[0]);
  f.backward.input_bytes = sizeof(in);
  f.backward.outputs = out;
  f.backward.output_stride = sizeof(out[0]);
  f.backward.output_bytes = sizeof(out);
  memset(f.gradients, 0, sizeof(f.gradients));
  dispatch(runtime, &f.backward);
  CHECK(memcmp(expected, f.gradients, sizeof(expected)) == 0);
  CHECK(in[0].extension == UINT64_MAX && out[0].extension == UINT64_MAX);
}

static uint32_t float_bits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static void check_metadata_aliasing(const et_kernel_runtime *runtime) {
  fixture f;
  elementwise_fixture(&f, swi_rows[0], 1);
  seed_values(&f);
  et_kernel_error error;
  const et_kernel_provider_v1 *provider = et_mod5_kernel_provider_v1();

  /* Point swiglu.forward output at its own writable shape array. */
  uint64_t writable_shape[3] = {1u, 1u, 1u};
  f.output.shape = writable_shape;
  f.output.data = writable_shape;
  CHECK(provider->validate_call(&f.forward, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(error.code == ET_KERNEL_CODE_ALIASING_OUTPUT);
  CHECK(et_kernel_runtime_dispatch(runtime, &f.forward, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(error.code == ET_KERNEL_CODE_ALIASING_OUTPUT);
  CHECK(writable_shape[0] == 1u && writable_shape[1] == 1u &&
        writable_shape[2] == 1u);

  /* Point swiglu.backward output at the live outputs table, request shape, and error. */
  elementwise_fixture(&f, swi_rows[0], 1);
  seed_values(&f);
  const float sentinel = f.gradients[1][0];
  et_kernel_tensor_view_v1 saved_grads[3];
  memcpy(saved_grads, f.grads, sizeof(saved_grads));
  f.grads[0].data = &f.grads[1];
  CHECK(provider->validate_call(&f.backward, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(error.code == ET_KERNEL_CODE_ALIASING_OUTPUT);
  CHECK(et_kernel_runtime_dispatch(runtime, &f.backward, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(error.code == ET_KERNEL_CODE_ALIASING_OUTPUT);
  CHECK(f.gradients[1][0] == sentinel);
  CHECK(memcmp(&f.grads[1], &saved_grads[1], sizeof(saved_grads[1])) == 0);

  f.grads[0].data = f.gradients[0];
  uint64_t req_shape[3] = {1u, 1u, 1u};
  f.backward_request.shape = req_shape;
  f.grads[1].data = req_shape;
  CHECK(provider->validate_call(&f.backward, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(error.code == ET_KERNEL_CODE_ALIASING_OUTPUT);
  CHECK(req_shape[0] == 1u && req_shape[1] == 1u && req_shape[2] == 1u);

  f.grads[1].data = &error;
  CHECK(provider->validate_call(&f.backward, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(error.code == ET_KERNEL_CODE_ALIASING_OUTPUT);
}

static void check_attention_accumulation_order(const et_kernel_runtime *runtime) {
  /* Cancellation-sensitive MQA [1,4,1,3,3,2] fixture verifying exact bitwise
     dQ (increasing tk) and shared dK/dV (query-head-major, then query-position)
     accumulation order. */
  fixture f;
  attention_fixture(&f, attention_rows[1], 5);
  memset(f.operands, 0, sizeof(f.operands));
  memset(f.upstream, 0, sizeof(f.upstream));
  memset(f.slopes, 0, sizeof(f.slopes));
  f.query_pos[0] = 2;
  f.query_pos[1] = 3;
  f.query_pos[2] = 4;
  f.key_pos[0] = 0;
  f.key_pos[1] = 1;
  f.key_pos[2] = 2;
  /* Admit keys {0,1} for tq=0,1 (p = 0.5 each) and {0,1,2} for tq=2 (p = 1/3). */
  memset(f.mask, 0, sizeof(f.mask));
  f.mask[0] = 1u; f.mask[1] = 1u;
  f.mask[3] = 1u; f.mask[4] = 1u;
  f.mask[6] = 1u; f.mask[7] = 1u; f.mask[8] = 1u;

  /* Q uses d=1 only and K uses d=0 only, so Q.K == 0 for every pair and
     probabilities are exact uniform 1/2 (tq=0,1) and 1/3 (tq=2). */
  const float big = 16777216.0f; /* 2^24 */
  /* V[n=0, hkv=0, tk, d=0]: (3, -1, -2) => tq=0,1: dp=(3,-1), mean=1, ds0=1.0;
     tq=2: dp=(3,-1,-2), mean=0, ds=(3p, -p, -2p). */
  f.operands[2][0 * 2 + 0] = 3.0f;
  f.operands[2][1 * 2 + 0] = -1.0f;
  f.operands[2][2 * 2 + 0] = -2.0f;

  /* K[n=0, hkv=0, tk, d=0] triggers exact cancellation on dQ[0, hq=0, tq=2, d=0]:
     tk=0: +3p*scale*big, tk=1: -3p*scale*big, tk=2: +p*scale. */
  f.operands[1][0 * 2 + 0] = big;
  f.operands[1][1 * 2 + 0] = 3.0f * big;
  f.operands[1][2 * 2 + 0] = -0.5f;

  /* Q[n=0, hq, tq, d=1] triggers exact cancellation on dK[0, hkv=0, tk=0, d=1]:
     (hq=0,tq=0)=+big, (hq=0,tq=1)=-big, (hq=1,tq=0)=+1.0. */
  f.operands[0][(0u * 3u + 0u) * 2u + 1u] = big;
  f.operands[0][(0u * 3u + 1u) * 2u + 1u] = -big;
  f.operands[0][(1u * 3u + 0u) * 2u + 1u] = 1.0f;

  /* upstream[n=0, hq, tq, d=0] = 1.0 for active rows;
     upstream[n=0, hq, tq, d=1] triggers exact cancellation on dV[0, hkv=0, tk=0, d=1]:
     (hq=0,tq=0)=+2*big, (hq=0,tq=1)=-2*big, (hq=1,tq=0)=+1.0. */
  f.upstream[(0u * 3u + 0u) * 2u + 0u] = 1.0f;
  f.upstream[(0u * 3u + 1u) * 2u + 0u] = 1.0f;
  f.upstream[(0u * 3u + 2u) * 2u + 0u] = 1.0f;
  f.upstream[(1u * 3u + 0u) * 2u + 0u] = 1.0f;
  f.upstream[(0u * 3u + 0u) * 2u + 1u] = 2.0f * big;
  f.upstream[(0u * 3u + 1u) * 2u + 1u] = -2.0f * big;
  f.upstream[(1u * 3u + 0u) * 2u + 1u] = 1.0f;

  dispatch(runtime, &f.backward);

  const float scale = 1.0f / sqrtf(2.0f);
  const float p3 = 1.0f / 3.0f;
  const float mean3 =
      ((0.0f + p3 * 3.0f) + p3 * (-1.0f)) + p3 * (-2.0f);
  const float t0 = ((p3 * (3.0f - mean3)) * scale) * big;
  const float t1 = ((p3 * (-1.0f - mean3)) * scale) * (3.0f * big);
  const float t2 = ((p3 * (-2.0f - mean3)) * scale) * (-0.5f);
  const float dq_expected = ((0.0f + t0) + t1) + t2;
  const float dq_reversed = ((0.0f + t2) + t1) + t0;
  CHECK(float_bits(dq_expected) != float_bits(dq_reversed));
  CHECK(float_bits(f.gradients[0][(0u * 3u + 2u) * 2u + 0u]) ==
        float_bits(dq_expected));

  const float dk_head_major =
      (((0.0f + (1.0f * scale) * big) + (1.0f * scale) * (-big)) +
       (1.0f * scale) * 1.0f);
  const float dk_token_major =
      (((0.0f + (1.0f * scale) * big) + (1.0f * scale) * 1.0f) +
       (1.0f * scale) * (-big));
  CHECK(float_bits(dk_head_major) != float_bits(dk_token_major));
  CHECK(float_bits(f.gradients[1][0u * 2u + 1u]) == float_bits(dk_head_major));

  const float dv_head_major =
      (((0.0f + 0.5f * (2.0f * big)) + 0.5f * (-2.0f * big)) + 0.5f * 1.0f);
  const float dv_token_major =
      (((0.0f + 0.5f * (2.0f * big)) + 0.5f * 1.0f) + 0.5f * (-2.0f * big));
  CHECK(float_bits(dv_head_major) != float_bits(dv_token_major));
  CHECK(float_bits(f.gradients[2][0u * 2u + 1u]) == float_bits(dv_head_major));
}

int main(int argc, char **argv) {
  const int oracle = argc == 2 && strcmp(argv[1], "--oracle") == 0;
  CHECK(argc == 1 || oracle);
  et_kernel_runtime *runtime = discover_mod5();
  if (!oracle) {
    check_metadata(runtime);
    check_malformed(runtime);
    check_metadata_aliasing(runtime);
    check_numeric_errors(runtime);
    check_wide_numeric_errors(runtime);
    check_fenv(runtime);
    check_strided_tables(runtime);
    check_attention_accumulation_order(runtime);
  }
  check_cases(runtime, oracle);
  et_kernel_runtime_destroy(runtime);
  if (!oracle)
    puts("MOD5 conformance and primitive finite differences: PASS");
  return 0;
}
