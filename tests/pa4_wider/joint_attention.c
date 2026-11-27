#define _POSIX_C_SOURCE 200809L

#include "eshkol_transformer/a2_attention_abi.h"
#include "fixture.h"

#include <inttypes.h>
#include <time.h>

enum { A2 = PROVIDERS, ARMS, ROWS = 10, A2_MAX_ELEMENTS = 64 };
enum { ACTUAL_UPSTREAM = OUTPUTS, TENSORS };
enum { PAIRS = 12, WARMUP = 100, ITERATIONS = 1000, MASKED_ROW = 6 };

static const char *const names[ARMS] = {"reference", "online", "a2"};
static const char *const tensors[TENSORS] = {"O", "dQ", "dK", "dV", "dO"};
static fixture f;
static float coefficients[A2_MAX_ELEMENTS];
static float witness[ROWS][ARMS][TENSORS][A2_MAX_ELEMENTS];

static void prepare(const et_kernel_capability_v1 *capability, size_t row) {
  const et_kernel_shape_range_v1 *range = &capability->shape_ranges[row];
  const uint64_t bounds[] = {2, 4, 2, 3, 3, 4};
  uint64_t shape[6];
  require(range->rank == sizeof(shape) / sizeof(*shape), "A2 shape rank");
  for (size_t d = 0; d < range->rank; ++d) {
    const et_kernel_dimension_range_v1 *dim = &range->dimensions[d];
    require(!dim->maximum_unbounded && dim->minimum == dim->maximum &&
                dim->minimum > 0 && dim->minimum <= bounds[d],
            "A2 shape is outside the bounded singleton mix");
    shape[d] = dim->minimum;
  }
  require(shape[0] * shape[1] * shape[3] * shape[5] <= A2_MAX_ELEMENTS &&
              shape[0] * shape[2] * shape[4] * shape[5] <= A2_MAX_ELEMENTS,
          "A2 shape exceeds witness storage");
  init_fixture(&f, shape);
  for (size_t i = 0; i < f.q_count; ++i)
    coefficients[i] = sample_value(i, 101u);
  if (row == MASKED_ROW) {
    require(shape[3] == 3 && shape[4] == 3, "expected masked-row shape");
    memset(f.mask + 2u * shape[4], 0, shape[4]);
  }
}

static size_t tensor_bytes(size_t tensor) {
  return tensor == ACTUAL_UPSTREAM ? f.in[UPSTREAM].byte_length
                                   : f.out[tensor].byte_length;
}

static float *tensor_data(size_t tensor) {
  return tensor == ACTUAL_UPSTREAM ? f.upstream : f.result[tensor];
}

static void poison_outputs(void) {
  for (size_t t = 0; t < TENSORS; ++t)
    for (size_t i = 0; i < tensor_bytes(t) / sizeof(float); ++i)
      tensor_data(t)[i] = NAN;
}

static void derive_upstream(void) {
  /* d/dO of sum_i (O[i]^2 / 4 + coeff[i] * O[i]). Each arm uses its own O. */
  for (size_t i = 0; i < f.q_count; ++i)
    f.upstream[i] = 0.5f * f.result[OUTPUT][i] + coefficients[i];
}

static void joint(const et_kernel_runtime *runtime) {
  dispatch(runtime, &f.call[FORWARD]);
  derive_upstream();
  dispatch(runtime, &f.call[BACKWARD]);
}

static void check_exact(size_t row, int arm, size_t first, size_t last) {
  for (size_t t = first; t <= last; ++t)
    require(memcmp(tensor_data(t), witness[row][arm][t], tensor_bytes(t)) == 0,
            "tensor differs from its certified bits");
}

static void check_bad_upstream(size_t row, int arm,
                               const et_kernel_runtime *runtime) {
  const size_t masked = 2u * f.shape[5];
  const size_t head_stride = f.shape[3] * f.shape[5];
  for (size_t h = 0; h < f.shape[1]; ++h)
    for (size_t d = 0; d < f.shape[5]; ++d) {
      const size_t i = h * head_stride + masked + d;
      const float o = f.result[OUTPUT][i], dq = f.result[DQ][i];
      require(o == 0.0f && !signbit(o) && dq == 0.0f && !signbit(dq),
              "empty row must have positive-zero output and dQ");
    }
  f.upstream[masked] = NAN;
  et_kernel_error error;
  require(et_kernel_runtime_dispatch(runtime, &f.call[BACKWARD], &error) ==
                  ET_KERNEL_ERROR_INVALID_ARGUMENT &&
              error.category == ET_KERNEL_ERROR_INVALID_ARGUMENT &&
              error.code == ET_KERNEL_CODE_INVALID_BUFFER &&
              strcmp(error.operation, "causal-attention.backward") == 0,
          "backward must reject nonfinite generated dO even on an empty row");
  check_exact(row, arm, OUTPUT, DV);
  derive_upstream();
  dispatch(runtime, &f.call[BACKWARD]);
  check_exact(row, arm, OUTPUT, ACTUAL_UPSTREAM);
}

static void capture(size_t row, int arm, const et_kernel_runtime *runtime) {
  poison_outputs();
  joint(runtime);
  for (size_t t = 0; t < TENSORS; ++t) {
    const size_t bytes = tensor_bytes(t);
    int nonzero = 0;
    for (size_t i = 0; i < bytes / sizeof(float); ++i) {
      const float actual = tensor_data(t)[i];
      require(isfinite(actual), "nonfinite or unwritten witness");
      nonzero |= actual != 0.0f;
    }
    if (t == OUTPUT || t == DV || t == ACTUAL_UPSTREAM)
      require(nonzero, "unexercised output, dO or dV");
    memcpy(witness[row][arm][t], tensor_data(t), bytes);
  }
  if (row == MASKED_ROW)
    check_bad_upstream(row, arm, runtime);
  if (row == 0) {
    /* One query and key per head. O = V and dV = dO. */
    require(memcmp(f.result[OUTPUT], f.v, tensor_bytes(OUTPUT)) == 0 &&
                memcmp(f.result[DV], f.upstream, tensor_bytes(DV)) == 0,
            "single-key forward/backward identity");
  }
}

static void certify(size_t row, et_kernel_runtime *const *runtimes) {
  capture(row, A2, runtimes[A2]);
  capture(row, REF, runtimes[REF]);
  check_exact(row, A2, OUTPUT, ACTUAL_UPSTREAM);
  capture(row, ONLINE, runtimes[ONLINE]);

  for (size_t t = 0; t < TENSORS; ++t) {
    error_metrics metrics = {0};
    size_t differences = 0;
    const size_t words = tensor_bytes(t) / sizeof(float);
    for (size_t i = 0; i < words; ++i) {
      const float *a2 = &witness[row][A2][t][i];
      const float *online = &witness[row][ONLINE][t][i];
      differences += memcmp(a2, online, sizeof(float)) != 0;
      require(compare(*a2, *online, &metrics), "online joint tolerance");
    }
    printf("BITS %zu %s words=%zu reference_different=0 online_different=%zu "
           "online_abs=%.9g online_rel=%.9g online_ulps=%" PRIu32 "\n",
           row, tensors[t], words, differences, metrics.absolute,
           metrics.relative, metrics.ulps);
  }

  /* Separate backward arithmetic from changes that arrived through online O. */
  memcpy(f.upstream, witness[row][A2][ACTUAL_UPSTREAM],
         tensor_bytes(ACTUAL_UPSTREAM));
  dispatch(runtimes[ONLINE], &f.call[BACKWARD]);
  check_exact(row, A2, DQ, DV);
}

static uint64_t now_ns(void) {
  struct timespec now;
  require(clock_gettime(CLOCK_MONOTONIC, &now) == 0, "read monotonic clock");
  return (uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec;
}

static uint64_t time_batch(size_t row, int arm,
                           const et_kernel_runtime *runtime) {
  for (int i = 0; i < WARMUP; ++i)
    joint(runtime);
  poison_outputs();
  const uint64_t start = now_ns();
  for (int i = 0; i < ITERATIONS; ++i)
    joint(runtime);
  const uint64_t elapsed = now_ns() - start;
  require(elapsed > 0, "nonpositive batch duration");
  check_exact(row, arm, OUTPUT, ACTUAL_UPSTREAM);
  return elapsed;
}

static void measure(size_t row, int arm, et_kernel_runtime *const *runtimes,
                    uint64_t totals[2]) {
  uint64_t sum[2] = {0};
  double minimum[2] = {INFINITY, INFINITY}, maximum[2] = {0};
  for (int pair = 0; pair < PAIRS; ++pair) {
    const int order = pair % 2;
    const int first = order ? arm : A2;
    const int second = order ? A2 : arm;
    uint64_t ns[2];
    ns[order] = time_batch(row, first, runtimes[first]);
    ns[1 - order] = time_batch(row, second, runtimes[second]);
    const double ratio = (double)ns[1] / (double)ns[0];
    minimum[order] = fmin(minimum[order], ratio);
    maximum[order] = fmax(maximum[order], ratio);
    sum[0] += ns[0];
    sum[1] += ns[1];
    printf("SAMPLE %zu %s %d %s %" PRIu64 " %" PRIu64 "\n", row, names[arm],
           pair, order ? "BA" : "AB", ns[0], ns[1]);
  }
  totals[0] += sum[0];
  totals[1] += sum[1];
  printf("RATIO %zu %s total=%.6f AB[%.6f,%.6f] BA[%.6f,%.6f]\n", row,
         names[arm], (double)sum[1] / (double)sum[0], minimum[0], maximum[0],
         minimum[1], maximum[1]);
}

int main(int argc, char **argv) {
  const int timing = argc == 2 && strcmp(argv[1], "--time") == 0;
  require(argc == 1 || timing, "usage: joint-attention [--time]");
  et_kernel_runtime *runtimes[ARMS] = {
      [A2] = discover_provider(et_a2_kernel_provider_v1()),
      [REF] = discover(REF),
      [ONLINE] = discover(ONLINE)};
  const et_kernel_capability_v1 *capability = et_kernel_runtime_capability_find(
      runtimes[A2], "kernel.causal-attention");
  require(capability != NULL && capability->shape_range_count == ROWS,
          "expected ten admitted A2 attention shapes");
  for (size_t row = 0; row < ROWS; ++row) {
    prepare(capability, row);
    printf("SHAPE %zu [%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
           ",%" PRIu64 "]\n",
           row, f.shape[0], f.shape[1], f.shape[2], f.shape[3], f.shape[4],
           f.shape[5]);
    certify(row, runtimes);
  }
  puts("PA4-Wider joint conformance: PASS "
       "(ten shapes, three failure/retry cases)");

  if (timing) {
    uint64_t totals[PROVIDERS][2] = {{0}};
    printf("TIMING clock=monotonic scope=joint-k1-dispatch pairs=%d warmup=%d "
           "iterations=%d sample_ns=a2,pa4-wider\n",
           PAIRS, WARMUP, ITERATIONS);
    for (size_t row = 0; row < ROWS; ++row) {
      prepare(capability, row);
      for (int arm = REF; arm <= ONLINE; ++arm)
        measure(row, arm, runtimes, totals[arm]);
    }
    for (int arm = REF; arm <= ONLINE; ++arm)
      printf("TOTAL %s a2_ns=%" PRIu64 " wider_ns=%" PRIu64 " ratio=%.6f\n",
             names[arm], totals[arm][0], totals[arm][1],
             (double)totals[arm][1] / (double)totals[arm][0]);
    printf("PA4-Wider joint timing: PASS (%d pairs, %d batches verified)\n",
           ROWS * PROVIDERS * PAIRS, 2 * ROWS * PROVIDERS * PAIRS);
  }
  for (int arm = REF; arm < ARMS; ++arm)
    et_kernel_runtime_destroy(runtimes[arm]);
  return EXIT_SUCCESS;
}
