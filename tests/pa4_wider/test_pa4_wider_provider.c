#include "fixture.h"

#include <errno.h>
#include <fenv.h>
#include <float.h>
#include <inttypes.h>
#include <xmmintrin.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

static fixture f;
static float reference[OUTPUTS][MAX_ELEMENTS];
static double oracle[OUTPUTS][MAX_ELEMENTS];
static size_t comparisons, gradient_checks;

static void compute_oracle(void) {
  const size_t n = f.shape[0], hq = f.shape[1], hkv = f.shape[2];
  const size_t tq = f.shape[3], tk = f.shape[4], dh = f.shape[5];
  const double scale = 1.0 / sqrt((double)dh);
  memset(oracle, 0, sizeof(oracle));
  for (size_t b = 0; b < n; ++b) {
    for (size_t h = 0; h < hq; ++h) {
      const size_t kv = (b * hkv + h / (hq / hkv)) * tk * dh;
      for (size_t i = 0; i < tq; ++i) {
        const size_t q = ((b * hq + h) * tq + i) * dh;
        double weights[MAX_TOKENS] = {0.0};
        double sum = 0.0;
        /* Fixture scores are bounded, so direct exponentiation is safe in f64.
         */
        for (size_t j = 0; j < tk; ++j) {
          if (f.mask[(b * tq + i) * tk + j] == 0u ||
              f.kp[b * tk + j] > f.qp[b * tq + i])
            continue;
          double dot = 0.0;
          for (size_t d = 0; d < dh; ++d)
            dot += (double)f.q[q + d] * f.k[kv + j * dh + d];
          weights[j] = exp(dot * scale);
          sum += weights[j];
        }
        if (sum == 0.0)
          continue;
        for (size_t j = 0; j < tk; ++j)
          for (size_t d = 0; d < dh; ++d)
            oracle[OUTPUT][q + d] += weights[j] / sum * f.v[kv + j * dh + d];

        for (size_t j = 0; j < tk; ++j) {
          const size_t key = kv + j * dh;
          const double p = weights[j] / sum;
          double centered = 0.0;
          for (size_t d = 0; d < dh; ++d)
            centered += (double)f.upstream[q + d] *
                        (f.v[key + d] - oracle[OUTPUT][q + d]);
          for (size_t d = 0; d < dh; ++d) {
            oracle[DQ][q + d] += p * centered * scale * f.k[key + d];
            oracle[DK][key + d] += p * centered * scale * f.q[q + d];
            oracle[DV][key + d] += p * f.upstream[q + d];
          }
        }
      }
    }
  }
}

static void parity(et_kernel_runtime *const *runtimes, const uint64_t *shape) {
  init_fixture(&f, shape);
  compute_oracle();
  error_metrics metrics[OUTPUTS] = {{0}};
  double oracle_abs[PROVIDERS] = {0.0};
  for (int p = REF; p <= ONLINE; ++p) {
    dispatch(runtimes[p], &f.call[FORWARD]);
    dispatch(runtimes[p], &f.call[BACKWARD]);
    for (size_t j = 0; j < OUTPUTS; ++j) {
      const size_t count = f.out[j].byte_length / sizeof(float);
      for (size_t i = 0; i < count; ++i) {
        const float actual = f.result[j][i];
        const double expected = oracle[j][i];
        const double delta = fabs(actual - expected);
        oracle_abs[p] = fmax(oracle_abs[p], delta);
        require(delta <= 2e-6 + 6e-6 * fabs(expected), "f64 oracle comparison");
        if (p == REF) {
          reference[j][i] = actual;
        } else {
          require(compare(reference[j][i], actual, &metrics[j]),
                  "reference/online parity");
          if (j != OUTPUT)
            require(memcmp(&reference[j][i], &actual, sizeof(actual)) == 0,
                    "backward parity preserves reference arithmetic");
          ++comparisons;
        }
      }
    }
  }
  printf("PARITY %" PRIu64 "x%" PRIu64 "x%" PRIu64 "x%" PRIu64 "x%" PRIu64
         "x%" PRIu64 " oracle_abs(ref,online)=%.17g,%.17g\n",
         shape[0], shape[1], shape[2], shape[3], shape[4], shape[5],
         oracle_abs[REF], oracle_abs[ONLINE]);
  static const char *const names[] = {"O", "dQ", "dK", "dV"};
  for (size_t j = 0; j < OUTPUTS; ++j)
    printf("  %s abs=%.17g rel=%.17g ulps=%" PRIu32 "\n", names[j],
           metrics[j].absolute, metrics[j].relative, metrics[j].ulps);
}

static double objective(const et_kernel_runtime *runtime) {
  dispatch(runtime, &f.call[FORWARD]);
  double loss = 0.0;
  for (size_t i = 0; i < f.q_count; ++i)
    loss += (double)f.result[OUTPUT][i] * f.upstream[i];
  return loss;
}

static double centered_difference(const et_kernel_runtime *runtime,
                                  float *value, float step) {
  const float original = *value;
  *value = original + step;
  const double plus = objective(runtime);
  *value = original - step;
  const double minus = objective(runtime);
  *value = original;
  return (plus - minus) / (2.0 * step);
}

static double check_gradients(const et_kernel_runtime *runtime,
                              int exhaustive) {
  float *const inputs[] = {f.q, f.k, f.v};
  double max_abs = 0.0;
  dispatch(runtime, &f.call[BACKWARD]);
  for (size_t tensor = 0; tensor < 3u; ++tensor) {
    const size_t count = tensor == 0u ? f.q_count : f.kv_count;
    const size_t coordinates = exhaustive ? count : 20u;
    size_t nonzero = 0u;
    for (size_t j = 0; j < coordinates; ++j) {
      const size_t i = exhaustive ? j : (count - 1u) * j / (coordinates - 1u);
      float *value = inputs[tensor] + i;
      /* Two central stencils cancel the O(h^2) term while keeping f32
         rounding noise below that of a tiny perturbation. */
      const double coarse = centered_difference(runtime, value, 0x1p-4f);
      const double fine = centered_difference(runtime, value, 0x1p-5f);
      const double numeric = (4.0 * fine - coarse) / 3.0;
      const double analytic = f.result[DQ + tensor][i];
      const double delta = fabs(numeric - analytic);
      max_abs = fmax(max_abs, delta);
      if (fabs(analytic) > 1e-5)
        ++nonzero;
      if (delta > 4e-5 + 3e-4 * fabs(analytic))
        fprintf(stderr,
                "gradient tensor=%zu coordinate=%zu numerical=%.12g "
                "analytical=%.12g error=%.12g\n",
                tensor, i, numeric, analytic, delta);
      require(delta <= 4e-5 + 3e-4 * fabs(analytic),
              "central gradient difference");
      ++gradient_checks;
    }
    require(nonzero >= 2u, "finite differences include active coordinates");
  }
  return max_abs;
}

static void reset_sentinels(void) { memset(f.result, 0xa5, sizeof(f.result)); }

static void expect_sentinels(void) {
  require(memcmp(f.result, reference, sizeof(f.result)) == 0,
          "failure or validation modified output");
}

static void rejected(const et_kernel_provider_v1 *p, et_kernel_runtime *runtime,
                     et_kernel_call_v1 *call, et_kernel_error_code code,
                     int k1) {
  et_kernel_error error;
  reset_sentinels();
  const int32_t status = p->validate_call(call, &error);
  require(status > 0 && status == (int32_t)error.category &&
              error.code == code && error.operation[0] != '\0' &&
              error.message[0] != '\0',
          "structured direct validation rejection");
  expect_sentinels();
  if (k1) {
    require(et_kernel_runtime_dispatch(runtime, call, &error) != 0,
            "K1 rejects malformed call");
    expect_sentinels();
  }
}

static void metadata_rejected(const et_kernel_provider_v1 *p,
                              et_kernel_runtime *runtime, void *metadata,
                              size_t bytes) {
  require(bytes <= 512u, "test snapshot fits");
  unsigned char snapshot[512];
  f.out[OUTPUT].data = metadata;
  memcpy(snapshot, metadata, bytes);
  rejected(p, runtime, &f.call[FORWARD], ET_KERNEL_CODE_ALIASING_OUTPUT, 1);
  require(memcmp(metadata, snapshot, bytes) == 0,
          "failure modified borrowed metadata");
  f.out[OUTPUT].data = f.result[OUTPUT];
}

static void structural_cases(const et_kernel_provider_v1 *p,
                             et_kernel_runtime *runtime) {
  const uint64_t small[] = {1, 2, 1, 2, 2, 2};
  init_fixture(&f, small);
  et_kernel_call_v1 *const call = &f.call[FORWARD];
  et_kernel_request_v1 *const request = &f.request[FORWARD];
  const et_kernel_call_v1 good_call = *call;
  const et_kernel_request_v1 good_request = *request;
  et_kernel_tensor_view_v1 *const output = &f.out[OUTPUT];
  const et_kernel_tensor_view_v1 good_output = *output;
  const et_kernel_tensor_view_v1 good_k = f.in[K];

  et_kernel_error error;
  require(p->validate_call(NULL, &error) != 0 &&
              error.code == ET_KERNEL_CODE_NULL_ARGUMENT,
          "null call rejected");
  call->request = NULL;
  rejected(p, runtime, call, ET_KERNEL_CODE_NULL_ARGUMENT, 1);
  *call = good_call;
  call->struct_size = sizeof(size_t);
  rejected(p, runtime, call, ET_KERNEL_CODE_INVALID_STRUCT_SIZE, 1);
  *call = good_call;
  request->struct_size = SIZE_MAX;
  rejected(p, runtime, call, ET_KERNEL_CODE_INVALID_STRUCT_SIZE, 0);
  *request = good_request;
  call->capability = "kernel.rope";
  rejected(p, runtime, call, ET_KERNEL_CODE_PROVIDER_REJECTED, 1);
  *call = good_call;
  request->operation = "causal-attention.other";
  rejected(p, runtime, call, ET_KERNEL_CODE_PROVIDER_REJECTED, 1);
  *request = good_request;
  request->deterministic = 2u;
  rejected(p, runtime, call, ET_KERNEL_CODE_INVALID_SHAPE, 1);
  *request = good_request;
  request->rank = 5u;
  rejected(p, runtime, call, ET_KERNEL_CODE_INVALID_SHAPE, 1);
  *request = good_request;
  request->shape = (const uint64_t *)((const unsigned char *)f.shape + 1);
  rejected(p, runtime, call, ET_KERNEL_CODE_INVALID_SHAPE, 1);
  *request = good_request;

  const uint64_t invalid[][6] = {
      {0, 2, 1, 2, 2, 2},  {3, 2, 1, 2, 2, 2},         {1, 0, 1, 2, 2, 2},
      {1, 5, 1, 2, 2, 2},  {1, 2, 0, 2, 2, 2},         {1, 3, 2, 2, 2, 2},
      {1, 2, 3, 2, 2, 2},  {1, 2, 1, 0, 2, 2},         {1, 2, 1, 65, 2, 2},
      {1, 2, 1, 2, 0, 2},  {1, 2, 1, 2, 65, 2},        {1, 2, 1, 2, 2, 0},
      {1, 2, 1, 2, 2, 17}, {1, 2, 1, 2, UINT64_MAX, 2}};
  for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i) {
    request->shape = invalid[i];
    rejected(p, runtime, call, ET_KERNEL_CODE_INVALID_SHAPE, 1);
  }
  *request = good_request;
  call->input_count--;
  rejected(p, runtime, call, ET_KERNEL_CODE_INVALID_BUFFER, 1);
  *call = good_call;
  et_kernel_tensor_view_v1 short_table[6];
  memcpy(short_table, f.in, sizeof(short_table));
  call->inputs = short_table;
  call->input_count = 8u;
  call->input_bytes = call->input_count * call->input_stride;
  /* The provider can reject an excessive count without inspecting entries. */
  rejected(p, runtime, call, ET_KERNEL_CODE_INVALID_BUFFER, 0);
  *call = good_call;
  call->output_stride--;
  rejected(p, runtime, call, ET_KERNEL_CODE_INVALID_STRUCT_SIZE, 1);
  *call = good_call;
  call->input_stride = SIZE_MAX - 7u;
  rejected(p, runtime, call, ET_KERNEL_CODE_INVALID_BUFFER, 1);
  *call = good_call;
  call->output_bytes--;
  rejected(p, runtime, call, ET_KERNEL_CODE_INVALID_BUFFER, 1);
  *call = good_call;
  call->outputs = (unsigned char *)f.out + 1;
  rejected(p, runtime, call, ET_KERNEL_CODE_INVALID_BUFFER, 1);
  *call = good_call;
  output->struct_size++;
  rejected(p, runtime, call, ET_KERNEL_CODE_INVALID_STRUCT_SIZE, 1);
  *output = good_output;
  f.in[K].struct_size = sizeof(size_t);
  f.in[K].dtype = (const char *)UINTPTR_MAX;
  rejected(p, runtime, call, ET_KERNEL_CODE_INVALID_STRUCT_SIZE, 1);
  f.in[K] = good_k;
  output->dtype = "i64";
  rejected(p, runtime, call, ET_KERNEL_CODE_INVALID_TEXT, 1);
  *output = good_output;
  output->device = "other";
  rejected(p, runtime, call, ET_KERNEL_CODE_INVALID_TEXT, 1);
  *output = good_output;
  output->offset_bytes = 8u;
  rejected(p, runtime, call, ET_KERNEL_CODE_INVALID_BUFFER, 1);
  *output = good_output;
  output->layout = 0u;
  rejected(p, runtime, call, ET_KERNEL_CODE_INVALID_BUFFER, 1);
  *output = good_output;
  output->rank--;
  rejected(p, runtime, call, ET_KERNEL_CODE_INVALID_SHAPE, 1);
  *output = good_output;
  output->shape = f.kv_shape;
  rejected(p, runtime, call, ET_KERNEL_CODE_INVALID_SHAPE, 1);
  *output = good_output;
  output->byte_length--;
  rejected(p, runtime, call, ET_KERNEL_CODE_INVALID_BUFFER, 1);
  *output = good_output;
  output->data = (unsigned char *)f.result[OUTPUT] + 1;
  rejected(p, runtime, call, ET_KERNEL_CODE_INVALID_BUFFER, 1);
  *output = good_output;
  output->data = (void *)(UINTPTR_MAX - 3u);
  rejected(p, runtime, call, ET_KERNEL_CODE_INVALID_BUFFER, 1);
  *output = good_output;
  output->data = f.q + 1;
  rejected(p, runtime, call, ET_KERNEL_CODE_ALIASING_OUTPUT, 1);
  *output = good_output;
  f.in[K].data = f.in[V].data;
  rejected(p, runtime, call, ET_KERNEL_CODE_ALIASING_OUTPUT, 1);
  f.in[K] = good_k;
  f.out[DV].data = f.result[DK];
  rejected(p, runtime, &f.call[BACKWARD], ET_KERNEL_CODE_ALIASING_OUTPUT, 1);
  f.out[DV].data = f.result[DV];

  metadata_rejected(p, runtime, call, sizeof(*call));
  metadata_rejected(p, runtime, request, sizeof(*request));
  metadata_rejected(p, runtime, f.in, sizeof(f.in));
  metadata_rejected(p, runtime, f.out, sizeof(f.out));
  metadata_rejected(p, runtime, f.shape, sizeof(f.shape));
  metadata_rejected(p, runtime, f.kp_shape, sizeof(f.kp_shape));
  metadata_rejected(p, runtime, f.q_shape, sizeof(f.q_shape));
  _Alignas(float) char capability[32] = "kernel.causal-attention";
  call->capability = capability;
  metadata_rejected(p, runtime, capability, sizeof(capability));
  *call = good_call;
  _Alignas(float) char operation[32] = "causal-attention.forward";
  request->operation = operation;
  metadata_rejected(p, runtime, operation, sizeof(operation));
  *request = good_request;
  _Alignas(float) char f32[32] = "f32", cpu[32] = "cpu";
  request->dtype = f32;
  metadata_rejected(p, runtime, f32, sizeof(f32));
  *request = good_request;
  request->device = cpu;
  metadata_rejected(p, runtime, cpu, sizeof(cpu));
  *request = good_request;
  f.in[K].dtype = f32;
  metadata_rejected(p, runtime, f32, sizeof(f32));
  f.in[K] = good_k;
  f.in[K].device = cpu;
  metadata_rejected(p, runtime, cpu, sizeof(cpu));
  f.in[K] = good_k;
  output->dtype = f32;
  metadata_rejected(p, runtime, f32, sizeof(f32));
  *output = good_output;
  output->device = cpu;
  metadata_rejected(p, runtime, cpu, sizeof(cpu));
  *output = good_output;

  /* Future descriptor tails and per-entry table padding are borrowed, too. */
  struct extended_call {
    et_kernel_call_v1 prefix;
    uint64_t tail[4];
  };
  struct extended_request {
    et_kernel_request_v1 prefix;
    uint64_t tail[4];
  };
  struct extended_view {
    et_kernel_tensor_view_v1 prefix;
    uint64_t tail[4];
  };
  struct extended_call extra_call = {good_call, {0}};
  struct extended_request extra_request = {good_request, {0}};
  struct extended_view extra_inputs[6] = {0};
  extra_call.prefix.struct_size = sizeof(extra_call);
  extra_request.prefix.struct_size = sizeof(extra_request);
  extra_call.prefix.request = &extra_request.prefix;
  for (size_t i = 0; i < 6u; ++i) {
    extra_inputs[i].prefix = f.in[i];
    extra_inputs[i].prefix.struct_size = sizeof(extra_inputs[i]);
  }
  extra_call.prefix.inputs = extra_inputs;
  extra_call.prefix.input_stride = sizeof(extra_inputs[0]);
  extra_call.prefix.input_bytes = sizeof(extra_inputs);
  reset_sentinels();
  require(p->validate_call(&extra_call.prefix, &error) == 0,
          "compatible appended metadata and table stride");
  require(error.code == ET_KERNEL_CODE_OK, "validation clears stale error");
  expect_sentinels();
  dispatch(runtime, &extra_call.prefix);
  void *const tails[] = {extra_call.tail, extra_request.tail,
                         extra_inputs[0].tail};
  for (size_t i = 0; i < sizeof(tails) / sizeof(tails[0]); ++i) {
    output->data = tails[i];
    rejected(p, runtime, &extra_call.prefix, ET_KERNEL_CODE_ALIASING_OUTPUT, 1);
  }
  *output = good_output;

  et_kernel_error aliased_error;
  memset(&aliased_error, 0xa5, sizeof(aliased_error));
  et_kernel_error saved_error;
  memcpy(&saved_error, &aliased_error, sizeof(saved_error));
  output->data = &aliased_error;
  require(p->validate_call(call, &aliased_error) != 0 &&
              memcmp(&saved_error, &aliased_error, sizeof(saved_error)) == 0,
          "direct diagnostic alias rejected without clobbering output");
  /* Malformed output table byte length must still preserve readable outputs. */
  call->output_bytes = call->output_stride + 8u;
  require(p->validate_call(call, &aliased_error) != 0 &&
              memcmp(&saved_error, &aliased_error, sizeof(saved_error)) == 0,
          "malformed output_bytes preserves aliased output");
  *call = good_call;
  /* Excessive output count must still preserve readable outputs. */
  et_kernel_tensor_view_v1 extra_out[4] = {good_output, good_output,
                                           good_output, good_output};
  extra_out[0].data = &aliased_error;
  call->outputs = extra_out;
  call->output_count = 4u;
  call->output_bytes = sizeof(extra_out);
  require(p->validate_call(call, &aliased_error) != 0 &&
              memcmp(&saved_error, &aliased_error, sizeof(saved_error)) == 0,
          "excessive output_count preserves aliased output 0");
  extra_out[0].data = f.result[OUTPUT];
  extra_out[3].data = &aliased_error;
  require(p->validate_call(call, &aliased_error) != 0 &&
              memcmp(&saved_error, &aliased_error, sizeof(saved_error)) == 0,
          "excessive output_count preserves aliased trailing output");
  *call = good_call;
  *output = good_output;
  require(p->validate_call(call, NULL) == 0,
          "direct validation permits null diagnostic");
  expect_sentinels();
}

static void numeric_cases(int variant, et_kernel_runtime *runtime) {
  const et_kernel_provider_v1 *p = provider(variant);
  init_fixture(&f, (const uint64_t[]){1, 1, 1, 2, 2, 1});
  et_kernel_call_v1 *const forward = &f.call[FORWARD];
  et_kernel_call_v1 *const backward = &f.call[BACKWARD];
  f.mask[0] = 2u;
  rejected(p, runtime, forward, ET_KERNEL_CODE_INVALID_BUFFER, 1);
  f.mask[0] = 1u;
  f.qp[0] = -1;
  rejected(p, runtime, forward, ET_KERNEL_CODE_INVALID_BUFFER, 1);
  f.qp[0] = f.qp[1];
  rejected(p, runtime, forward, ET_KERNEL_CODE_INVALID_BUFFER, 1);
  f.qp[0] = 1;
  f.kp[1] = ET_PA4_WIDER_MAX_EXACT_POSITION + INT64_C(1);
  rejected(p, runtime, backward, ET_KERNEL_CODE_INVALID_BUFFER, 1);
  f.kp[1] = ET_PA4_WIDER_MAX_EXACT_POSITION;
  dispatch(runtime, forward);
  f.kp[1] = 4;

  float *const vectors[] = {f.q, f.k, f.v, f.upstream};
  for (size_t i = 0; i < sizeof(vectors) / sizeof(vectors[0]); ++i) {
    float old = vectors[i][0];
    const uint32_t signaling_nan = UINT32_C(0x7f800001);
    memcpy(vectors[i], &signaling_nan, sizeof(signaling_nan));
    const int flags = fetestexcept(FE_ALL_EXCEPT);
    rejected(p, runtime, backward, ET_KERNEL_CODE_INVALID_BUFFER, 1);
    require(fetestexcept(FE_ALL_EXCEPT) == flags,
            "input scan preserves FP flags");
    vectors[i][0] = -INFINITY;
    rejected(p, runtime, backward, ET_KERNEL_CODE_INVALID_BUFFER, 1);
    vectors[i][0] = old;
  }
  memset(f.mask, 0, sizeof(f.mask));
  f.q[0] = f.k[0] = FLT_MAX;
  dispatch(runtime, forward);
  dispatch(runtime, backward);
  for (size_t t = 0; t < OUTPUTS; ++t)
    for (size_t i = 0; i < f.out[t].byte_length / sizeof(float); ++i)
      require(f.result[t][i] == 0.0f && !signbit(f.result[t][i]),
              "fully masked produces positive zero");
  f.mask[0] = 1u;
  rejected(p, runtime, forward, ET_KERNEL_CODE_PROVIDER_REJECTED, 1);
  rejected(p, runtime, backward, ET_KERNEL_CODE_PROVIDER_REJECTED, 1);

  /* Each upstream contribution is finite, but their shared dV sum is not. */
  f.q[0] = f.q[1] = f.k[0] = f.k[1] = f.v[0] = f.v[1] = 0.0f;
  f.mask[2] = 1u;
  f.upstream[0] = f.upstream[1] = FLT_MAX * 0.75f;
  rejected(p, runtime, backward, ET_KERNEL_CODE_PROVIDER_REJECTED, 1);
  /* An admitted dP dot product overflows. */
  f.v[0] = 2.0f;
  rejected(p, runtime, backward, ET_KERNEL_CODE_PROVIDER_REJECTED, 1);

  f.v[0] = f.v[1] = FLT_MAX * 0.75f;
  f.mask[3] = 1u;
  if (variant == ONLINE) {
    rejected(p, runtime, forward, ET_KERNEL_CODE_PROVIDER_REJECTED, 1);
  } else {
    dispatch(runtime, forward);
    require(f.result[OUTPUT][1] == f.v[1],
            "normalized reference handles large V");
  }

  fenv_t saved;
  require(fegetenv(&saved) == 0, "snapshot test fenv");
  require(fesetround(FE_DOWNWARD) == 0, "set nondefault rounding");
  rejected(p, runtime, forward, ET_KERNEL_CODE_PROVIDER_REJECTED, 1);
  require(fegetround() == FE_DOWNWARD && fesetenv(&saved) == 0,
          "restore test rounding");
  const unsigned int mxcsr = _mm_getcsr();
  static const unsigned int changes[] = {0x0040u, 0x8000u, 0x2000u, 0x0080u};
  for (size_t i = 0; i < sizeof(changes) / sizeof(changes[0]); ++i) {
    _mm_setcsr(mxcsr ^ changes[i]);
    rejected(p, runtime, forward, ET_KERNEL_CODE_PROVIDER_REJECTED, 1);
    require(_mm_getcsr() == (mxcsr ^ changes[i]), "rejected MXCSR preserved");
    _mm_setcsr(mxcsr);
  }
  uint16_t original;
  __asm__ volatile("fnstcw %0" : "=m"(original));
  const uint16_t unmasked = original ^ UINT16_C(0x0020);
  __asm__ volatile("fldcw %0" : : "m"(unmasked));
  rejected(p, runtime, forward, ET_KERNEL_CODE_PROVIDER_REJECTED, 1);
  uint16_t current;
  __asm__ volatile("fnstcw %0" : "=m"(current));
  __asm__ volatile("fldcw %0" : : "m"(original));
  require(current == unmasked, "rejected x87 control word preserved");

  require(feclearexcept(FE_ALL_EXCEPT) == 0 && feraiseexcept(FE_DIVBYZERO) == 0,
          "set preexisting FP flag");
  const int flags = fetestexcept(FE_ALL_EXCEPT);
  const unsigned int captured_mxcsr = _mm_getcsr();
  f.v[0] = f.v[1] = 0.5f;
  f.q[1] = 1.0f;
  f.k[1] = 0x1p22f; /* exact dry run with expf underflow */
  errno = EDOM;
  et_kernel_error error;
  reset_sentinels();
  require(p->validate_call(forward, &error) == 0 && error.category == 0 &&
              errno == EDOM && fetestexcept(FE_ALL_EXCEPT) == flags &&
              _mm_getcsr() == captured_mxcsr,
          "successful dry run restores exception flags and errno");
  expect_sentinels();
  f.q[1] = FLT_MAX;
  rejected(p, runtime, forward, ET_KERNEL_CODE_PROVIDER_REJECTED, 0);
  require(errno == EDOM && fetestexcept(FE_ALL_EXCEPT) == flags &&
              _mm_getcsr() == captured_mxcsr,
          "failing dry run restores exception flags and errno");
  require(fesetenv(&saved) == 0, "restore test fenv");
}

static void capability_cases(int variant, et_kernel_runtime *runtime) {
  const et_kernel_provider_v1 *p = provider(variant);
  const et_kernel_capability_v1 *cap =
      et_kernel_runtime_capability_find(runtime, "kernel.causal-attention");
  require(p->abi_major == 1u && p->abi_minor == 0u &&
              p->required_features == 0u && p->capability_count == 1u &&
              cap != NULL && cap->deterministic == 1u &&
              cap->status == ET_KERNEL_CAPABILITY_VERIFIED &&
              cap->operation_count == 2u && cap->dtype_count == 1u &&
              cap->device_count == 1u && cap->shape_range_count == 8u &&
              strcmp(cap->implementation, p->name) == 0 &&
              strcmp(cap->dtypes[0], "f32") == 0 &&
              strcmp(cap->devices[0], "cpu") == 0,
          "K1 1.0 capability metadata");
  for (uint64_t hq = 1u; hq <= 4u; ++hq)
    for (uint64_t hkv = 1u; hkv <= 4u; ++hkv) {
      uint64_t shape[] = {2, hq, hkv, 64, 64, 16};
      const et_kernel_request_v1 request = {.struct_size = sizeof(request),
                                            .operation =
                                                "causal-attention.forward",
                                            .dtype = "f32",
                                            .device = "cpu",
                                            .rank = 6u,
                                            .shape = shape,
                                            .deterministic = 1u};
      et_kernel_error error;
      require((et_kernel_runtime_capability_require(
                   runtime, "kernel.causal-attention", &request, NULL,
                   &error) == 0) == (hq % hkv == 0u),
              "capability ranges match exact head pairs");
    }
}

int main(void) {
  et_kernel_runtime *runtimes[] = {discover(REF), discover(ONLINE)};
  memset(reference, 0xa5, sizeof(reference));
  for (int p = REF; p <= ONLINE; ++p) {
    capability_cases(p, runtimes[p]);
    structural_cases(provider(p), runtimes[p]);
    numeric_cases(p, runtimes[p]);
  }

  double grad_abs[PROVIDERS] = {0.0};
  for (size_t i = 0; i < sizeof(workload_shapes) / sizeof(workload_shapes[0]);
       ++i) {
    parity(runtimes, workload_shapes[i]);
    for (int p = REF; p <= ONLINE; ++p)
      grad_abs[p] = fmax(grad_abs[p], check_gradients(runtimes[p], 0));
  }
  parity(runtimes, (const uint64_t[]){2, 3, 3, 5, 7, 3});
  for (int p = REF; p <= ONLINE; ++p) {
    grad_abs[p] = fmax(grad_abs[p], check_gradients(runtimes[p], 1));
    et_kernel_runtime_destroy(runtimes[p]);
  }
  printf(
      "PA4-Wider conformance: PASS (%zu paired output comparisons, %zu central "
      "differences, gradient abs max ref=%.17g online=%.17g)\n",
      comparisons, gradient_checks, grad_abs[REF], grad_abs[ONLINE]);
  return EXIT_SUCCESS;
}
