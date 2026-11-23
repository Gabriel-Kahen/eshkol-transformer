#define _POSIX_C_SOURCE 200809L

#include "fixture.h"

#include <inttypes.h>
#include <time.h>

enum { PAIRS = 12, WARMUP = 15, ITERATIONS = 100 };
enum { DISPATCH, INVOKE };

static fixture f;
static float witness[PROVIDERS][OUTPUTS][MAX_ELEMENTS];

static uint64_t now_ns(void) {
  struct timespec now;
  require(clock_gettime(CLOCK_MONOTONIC, &now) == 0, "read monotonic clock");
  return (uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec;
}

static void run(const et_kernel_runtime *runtime,
                const et_kernel_provider_v1 *p, int operation, int scope,
                int iterations) {
  const int first = operation == BACKWARD ? BACKWARD : FORWARD;
  const int last = operation == FORWARD ? FORWARD : BACKWARD;
  if (scope == DISPATCH) {
    et_kernel_error error;
    for (int i = 0; i < iterations; ++i)
      for (int op = first; op <= last; ++op)
        require(et_kernel_runtime_dispatch(runtime, &f.call[op], &error) == 0,
                "benchmark dispatch");
  } else {
    for (int i = 0; i < iterations; ++i)
      for (int op = first; op <= last; ++op)
        p->invoke_call(&f.call[op]);
  }
}

static void zero_outputs(void) {
  for (size_t i = 0; i < OUTPUTS; ++i)
    memset(f.result[i], 0, f.out[i].byte_length);
}

static void build_witnesses(et_kernel_runtime *const *runtimes) {
  for (int p = REF; p <= ONLINE; ++p) {
    zero_outputs();
    for (int op = FORWARD; op <= BACKWARD; ++op) {
      et_kernel_error error;
      dispatch(runtimes[p], &f.call[op]);
      require(provider(p)->validate_call(&f.call[op], &error) == 0,
              "prevalidate immutable invoke benchmark");
    }
    for (size_t j = 0; j < OUTPUTS; ++j) {
      const size_t count = f.out[j].byte_length / sizeof(float);
      int nonzero = 0;
      error_metrics metrics = {0};
      for (size_t i = 0; i < count; ++i) {
        const float actual = f.result[j][i];
        require(isfinite(actual), "nonfinite baseline witness");
        nonzero |= actual != 0.0f;
        if (p == ONLINE)
          require(compare(witness[REF][j][i], actual, &metrics),
                  "reference/online benchmark witness mismatch");
        witness[p][j][i] = actual;
      }
      require(nonzero, "unexercised output witness");
    }
  }
}

static void verify_sample(int arm, int operation) {
  const int first = operation == BACKWARD ? DQ : OUTPUT;
  const int last = operation == FORWARD ? OUTPUT : DV;
  for (int j = first; j <= last; ++j) {
    require(memcmp(f.result[j], witness[arm][j], f.out[j].byte_length) == 0,
            "timed sample differs from its certified witness");
    const size_t count = f.out[j].byte_length / sizeof(float);
    error_metrics metrics = {0};
    for (size_t i = 0; i < count; ++i)
      require(compare(witness[REF][j][i], f.result[j][i], &metrics),
              "timed sample differs from reference");
  }
}

static uint64_t time_sample(const et_kernel_runtime *runtime, int arm,
                            int operation, int scope) {
  const et_kernel_provider_v1 *p = provider(arm);
  run(runtime, p, operation, scope, WARMUP);
  zero_outputs();
  const uint64_t start = now_ns();
  run(runtime, p, operation, scope, ITERATIONS);
  const uint64_t elapsed = now_ns() - start;
  require(elapsed > 0u, "nonpositive batch duration");
  verify_sample(arm, operation);
  return elapsed;
}

static int compare_times(const void *left, const void *right) {
  const uint64_t a = *(const uint64_t *)left, b = *(const uint64_t *)right;
  return (a > b) - (a < b);
}

static void benchmark_case(size_t shape_index, int operation, int scope,
                           et_kernel_runtime *const *runtimes) {
  static const char *const operations[] = {"forward", "backward", "joint"};
  static const char *const scopes[] = {"dispatch", "invoke"};
  uint64_t times[PROVIDERS][PAIRS];

  for (size_t pair = 0; pair < PAIRS; ++pair) {
    const int first = pair % 2u == 0u ? REF : ONLINE;
    const int second = 1 - first;
    times[first][pair] = time_sample(runtimes[first], first, operation, scope);
    times[second][pair] =
        time_sample(runtimes[second], second, operation, scope);
    printf("SAMPLE %zu %s %s %zu %s %" PRIu64 " %" PRIu64 "\n", shape_index,
           operations[operation], scopes[scope], pair,
           first == REF ? "AB" : "BA", times[REF][pair], times[ONLINE][pair]);
  }
  double min_ratio[2] = {INFINITY, INFINITY};
  double max_ratio[2] = {0.0, 0.0};
  for (size_t i = 0; i < PAIRS; ++i) {
    const double ratio = (double)times[ONLINE][i] / (double)times[REF][i];
    min_ratio[i % 2u] = fmin(min_ratio[i % 2u], ratio);
    max_ratio[i % 2u] = fmax(max_ratio[i % 2u], ratio);
  }
  for (int p = REF; p <= ONLINE; ++p)
    qsort(times[p], PAIRS, sizeof(times[p][0]), compare_times);
  const uint64_t ref = times[REF][(PAIRS - 1u) / 2u];
  const uint64_t online = times[ONLINE][(PAIRS - 1u) / 2u];
  printf("MEDIAN %zu %s %s %.2f %.2f %.6f AB[%.6f,%.6f] BA[%.6f,%.6f]\n",
         shape_index, operations[operation], scopes[scope],
         (double)ref / ITERATIONS, (double)online / ITERATIONS,
         (double)online / (double)ref, min_ratio[0], max_ratio[0], min_ratio[1],
         max_ratio[1]);
}

int main(void) {
  et_kernel_runtime *runtimes[] = {discover(REF), discover(ONLINE)};
  printf("PA4-Wider AB/BA pairs=%d warmup=%d iterations=%d "
         "sample_ns=per_batch median_ns=per_iteration A=reference B=online\n",
         PAIRS, WARMUP, ITERATIONS);
  for (size_t i = 0; i < sizeof(workload_shapes) / sizeof(workload_shapes[0]);
       ++i) {
    init_fixture(&f, workload_shapes[i]);
    printf("SHAPE %zu [%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
           ",%" PRIu64 "]\n",
           i, f.shape[0], f.shape[1], f.shape[2], f.shape[3], f.shape[4],
           f.shape[5]);
    build_witnesses(runtimes);
    for (int operation = FORWARD; operation <= JOINT; ++operation)
      for (int scope = DISPATCH; scope <= INVOKE; ++scope)
        benchmark_case(i, operation, scope, runtimes);
  }
  for (int p = REF; p <= ONLINE; ++p)
    et_kernel_runtime_destroy(runtimes[p]);
  puts("PA4-Wider benchmark: PASS (432 pairs, 864 timed samples verified)");
  return EXIT_SUCCESS;
}
