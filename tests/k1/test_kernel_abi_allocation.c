#include "eshkol_transformer/kernel_abi.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t fail_index = SIZE_MAX;
static size_t allocation_calls;
static size_t outstanding_allocations;
static int tracking;

void *__real_malloc(size_t size);
void *__real_calloc(size_t count, size_t size);
void __real_free(void *pointer);

void *__wrap_malloc(size_t size) {
  void *result;
  if (!tracking) {
    return __real_malloc(size);
  }
  if (allocation_calls++ == fail_index) {
    return NULL;
  }
  result = __real_malloc(size);
  if (result != NULL) {
    outstanding_allocations++;
  }
  return result;
}

void *__wrap_calloc(size_t count, size_t size) {
  void *result;
  if (!tracking) {
    return __real_calloc(count, size);
  }
  if (allocation_calls++ == fail_index) {
    return NULL;
  }
  result = __real_calloc(count, size);
  if (result != NULL) {
    outstanding_allocations++;
  }
  return result;
}

void __wrap_free(void *pointer) {
  if (tracking && pointer != NULL) {
    if (outstanding_allocations == 0u) {
      (void)fprintf(stderr, "free without tracked allocation\n");
      abort();
    }
    outstanding_allocations--;
  }
  __real_free(pointer);
}

static int checks;

#define CHECK(condition)                                                     \
  do {                                                                       \
    checks++;                                                                \
    if (!(condition)) {                                                      \
      (void)fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,        \
                    #condition);                                             \
      return 1;                                                              \
    }                                                                        \
  } while (0)

static int32_t validate_call(const et_kernel_call_v1 *call,
                             et_kernel_error *error) {
  (void)call;
  et_kernel_error_clear(error);
  return 0;
}

static void invoke_call(const et_kernel_call_v1 *call) { (void)call; }

static const char *const operations[] = {"storage.copy"};
static const char *const dtypes[] = {"f32"};
static const char *const devices[] = {"cpu"};
static const et_kernel_shape_range_v1 ranges[] = {
    {.rank = 0u, .dimensions = NULL},
};
static const et_kernel_capability_v1 capability = {
    .struct_size = sizeof(et_kernel_capability_v1),
    .name = "tensor.f32",
    .status = ET_KERNEL_CAPABILITY_VERIFIED,
    .implementation = "test-k1-allocation",
    .version = "1.0",
    .evidence = "TEST-ONLY:K1-allocation-failure-cleanup",
    .deterministic = 1u,
    .reserved = {0},
    .operation_count = 1u,
    .operations = operations,
    .dtype_count = 1u,
    .dtypes = dtypes,
    .device_count = 1u,
    .devices = devices,
    .shape_range_count = 1u,
    .shape_ranges = ranges,
};
static const et_kernel_provider_v1 provider = {
    .struct_size = sizeof(et_kernel_provider_v1),
    .abi_major = ET_KERNEL_ABI_MAJOR,
    .abi_minor = ET_KERNEL_ABI_MINOR,
    .required_features = 0u,
    .name = "test-k1-allocation",
    .version = "1.0",
    .evidence = "TEST-ONLY:K1-allocation-failure-cleanup",
    .capability_count = 1u,
    .capability_stride = sizeof(et_kernel_capability_v1),
    .capability_bytes = sizeof(et_kernel_capability_v1),
    .capabilities = &capability,
    .validate_call = validate_call,
    .invoke_call = invoke_call,
};

static const et_kernel_provider_v1 *resolve_provider(void *context,
                                                     const char *symbol) {
  checks += 3;
  if (context != &provider || symbol == NULL ||
      strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) != 0) {
    return NULL;
  }
  return &provider;
}

static void begin_tracking(size_t failure) {
  fail_index = failure;
  allocation_calls = 0u;
  outstanding_allocations = 0u;
  tracking = 1;
}

static int successful_round(size_t *calls) {
  et_kernel_error error;
  et_kernel_runtime *runtime = NULL;
  begin_tracking(SIZE_MAX);
  CHECK(et_kernel_runtime_discover(resolve_provider, (void *)&provider,
                                   &runtime, &error) == 0);
  CHECK(runtime != NULL);
  CHECK(error.category == ET_KERNEL_ERROR_NONE);
  *calls = allocation_calls;
  et_kernel_runtime_destroy(runtime);
  CHECK(outstanding_allocations == 0u);
  tracking = 0;
  return 0;
}

int main(void) {
  size_t total_sites = 0u;
  CHECK(successful_round(&total_sites) == 0);
  CHECK(total_sites > 2u);

  for (size_t site = 0u; site < total_sites; site++) {
    et_kernel_error error;
    et_kernel_runtime *runtime = (et_kernel_runtime *)(uintptr_t)1u;
    size_t retry_sites = 0u;
    begin_tracking(site);
    CHECK(et_kernel_runtime_discover(resolve_provider, (void *)&provider,
                                     &runtime, &error) ==
          ET_KERNEL_ERROR_INTERNAL);
    CHECK(runtime == NULL);
    CHECK(error.category == ET_KERNEL_ERROR_INTERNAL);
    CHECK(error.code == ET_KERNEL_CODE_ALLOCATION_FAILED);
    CHECK(outstanding_allocations == 0u);
    tracking = 0;

    CHECK(successful_round(&retry_sites) == 0);
    CHECK(retry_sites == total_sites);
  }

  (void)printf("K1 allocation PASS: %zu sites, %d checks\n", total_sites,
               checks);
  return 0;
}
