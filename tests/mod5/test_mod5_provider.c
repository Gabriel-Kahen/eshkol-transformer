#include "mod5_test.h"

#include <errno.h>
#include <fenv.h>
#include <float.h>
#include <xmmintrin.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

enum { ELEMENTS = 64 };

static const uint64_t rms_rows[][3] = {
    {1, 1, 1}, {1, 1, 4}, {1, 2, 4}, {2, 3, 4}, {2, 3, 8}};
static const uint64_t swi_rows[][3] = {
    {1, 1, 1}, {1, 1, 8}, {1, 2, 4}, {2, 3, 4}, {2, 3, 8}};
static const uint64_t attention_rows[][6] = {
    {1, 2, 1, 2, 2, 2}, {1, 4, 1, 3, 3, 2}, {2, 4, 1, 2, 3, 4},
    {1, 4, 2, 3, 3, 2}, {2, 4, 2, 2, 3, 4}, {1, 2, 2, 3, 3, 2}};

typedef struct fixture {
  et_kernel_request_v1 forward_request, backward_request;
  et_kernel_call_v1 forward, backward;
  et_kernel_tensor_view_v1 inputs[9], output, grads[3];
  uint64_t q[4], kv[4], qp[2], kp[2], mask_shape[3];
  float operands[3][ELEMENTS], upstream[ELEMENTS], slopes[4], epsilon;
  float result[ELEMENTS], gradients[3][ELEMENTS];
  int64_t query_pos[6], key_pos[6], window;
  uint8_t mask[18];
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
  f->window = mode == 1 ? 1 : mode == 3 ? INT64_MAX : 8;
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

static void check_repeat_and_gradients(const et_kernel_runtime *runtime,
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
  for (size_t field = 0; field < f->backward.output_count; ++field) {
    for (size_t i = 0; i < f->grads[field].byte_length / sizeof(float); ++i) {
      const float original = f->operands[field][i];
      f->operands[field][i] = original + step;
      const double plus = objective(runtime, f);
      f->operands[field][i] = original - step;
      const double minus = objective(runtime, f);
      f->operands[field][i] = original;
      check_close(f->gradients[field][i], (plus - minus) / (2.0 * step),
                  0.00045, 0.002);
    }
  }
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
  for (size_t i = 0; i < ARRAY_COUNT(rms_rows); ++i) {
    elementwise_fixture(&f, rms_rows[i], 0);
    seed_values(&f);
    if (oracle)
      dump_fixture(runtime, &f);
    else
      check_repeat_and_gradients(runtime, &f, 0);
  }
  for (size_t i = 0; i < ARRAY_COUNT(swi_rows); ++i) {
    elementwise_fixture(&f, swi_rows[i], 1);
    seed_values(&f);
    if (oracle)
      dump_fixture(runtime, &f);
    else
      check_repeat_and_gradients(runtime, &f, 0);
  }
  for (size_t i = 0; i < ARRAY_COUNT(attention_rows); ++i) {
    for (int mode = 0; mode < 5; ++mode) {
      attention_fixture(&f, attention_rows[i], mode);
      seed_values(&f);
      if (oracle)
        dump_fixture(runtime, &f);
      else
        check_repeat_and_gradients(runtime, &f, mode == 2);
    }
  }
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

int main(int argc, char **argv) {
  const int oracle = argc == 2 && strcmp(argv[1], "--oracle") == 0;
  CHECK(argc == 1 || oracle);
  et_kernel_runtime *runtime = discover_mod5();
  if (!oracle) {
    check_metadata(runtime);
    check_malformed(runtime);
    check_numeric_errors(runtime);
    check_fenv(runtime);
    check_strided_tables(runtime);
  }
  check_cases(runtime, oracle);
  et_kernel_runtime_destroy(runtime);
  if (!oracle)
    puts("MOD5 conformance and primitive finite differences: PASS");
  return 0;
}
