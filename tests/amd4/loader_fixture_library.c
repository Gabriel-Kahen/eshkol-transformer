#include "loader_fixture.h"

#include <assert.h>
#include <string.h>

#ifdef AMD4_FIXTURE_UNAVAILABLE
/* This relocation must fail under RTLD_NOW, even on a host with ROCm. */
extern int amd4_fixture_never_provided(void);

int amd4_fixture_unresolved(void) {
  return amd4_fixture_never_provided();
}
#else
/* Record device operations and their row or active ordinal across both DSOs. */
static void record(char operation, int32_t ordinal) {
  amd4_loader_fixture_state *state = &amd4_loader_fixture;
  assert(state->event_count + 2 < sizeof(state->events));
  assert(ordinal >= 0 && ordinal <= 1);
  state->events[state->event_count++] = operation;
  state->events[state->event_count++] = (char)('0' + ordinal);
}

#ifdef AMD4_FIXTURE_HIP
int32_t hipRuntimeGetVersion(int32_t *version) {
  *version = 60431234;
  return 0;
}

int32_t hipDriverGetVersion(int32_t *version) {
  *version = 60430000;
  return 0;
}

int32_t hipGetDeviceCount(int32_t *count) {
  *count = amd4_loader_fixture.count;
  return 0;
}

typedef struct hip_properties_r0600 hip_properties_r0600;

#ifdef AMD4_FIXTURE_MISSING_EXPORT
/* An unversioned export must never substitute for R0600. */
int32_t hipGetDeviceProperties(hip_properties_r0600 *properties,
                               int32_t ordinal) {
  (void)properties;
  record('x', ordinal);
  return 801;
}
#else
int32_t hipGetDevicePropertiesR0600(hip_properties_r0600 *properties,
                                    int32_t ordinal) {
  record('p', ordinal);
  /* HIP 6.4.3 hipDeviceProp_t LP64 layout. Write bytes independently of
     AMD4's private struct, including the full extent of the vendor type. */
  _Static_assert(sizeof(size_t) == 8, "fixture requires LP64 size_t");
  unsigned char *bytes = (unsigned char *)properties;
  const size_t total = UINT64_C(0x601234567) + (size_t)ordinal;
  const int32_t wavefront = 64, compute_units = 110 + ordinal;
  const int32_t bus = 83 + ordinal, device = 22, domain = 17;
  memset(bytes, 0xa5, 1472);
  memcpy(bytes, "fixture GPU", sizeof("fixture GPU"));
  memcpy(bytes + 288, &total, sizeof(total));
  memcpy(bytes + 308, &wavefront, sizeof(wavefront));
  memcpy(bytes + 388, &compute_units, sizeof(compute_units));
  memcpy(bytes + 584, &bus, sizeof(bus));
  memcpy(bytes + 588, &device, sizeof(device));
  memcpy(bytes + 592, &domain, sizeof(domain));
  memcpy(bytes + 1160, "gfx90a:xnack-", sizeof("gfx90a:xnack-"));
  return 0;
}
#endif

int32_t hipGetDevice(int32_t *ordinal) {
  record('g', amd4_loader_fixture.active);
  *ordinal = amd4_loader_fixture.active;
  return 0;
}

int32_t hipSetDevice(int32_t ordinal) {
  record('s', ordinal);
  if (ordinal == 1 && amd4_loader_fixture.restore_code != 0) {
    return amd4_loader_fixture.restore_code;
  }
  amd4_loader_fixture.active = ordinal;
  return 0;
}

int32_t hipMemGetInfo(size_t *free_bytes, size_t *total_bytes) {
  const int32_t active = amd4_loader_fixture.active;
  record('m', active);
  *free_bytes = active == 0 ? UINT64_C(0x401234567) : 0;
  *total_bytes = UINT64_C(0x501234567) + (size_t)active;
  return amd4_loader_fixture.memory_code;
}
#else
typedef struct _rocblas_handle {
  int32_t ordinal;
} *rocblas_handle;

static struct _rocblas_handle handle;
static int handle_open;

int32_t rocblas_get_version_string(char *buffer, size_t capacity) {
  assert(capacity == 256);
  memcpy(buffer, "fixture rocBLAS", sizeof("fixture rocBLAS"));
  return 0;
}

int32_t rocblas_create_handle(rocblas_handle *out) {
  record('c', amd4_loader_fixture.active);
  assert(!handle_open);
  handle_open = 1;
  handle.ordinal = amd4_loader_fixture.active;
  *out = &handle;
  return amd4_loader_fixture.create_code;
}

#ifndef AMD4_FIXTURE_MISSING_EXPORT
int32_t rocblas_destroy_handle(rocblas_handle received) {
  record('d', amd4_loader_fixture.active);
  assert(handle_open && received == &handle);
  assert(received->ordinal == amd4_loader_fixture.active);
  handle_open = 0;
  return amd4_loader_fixture.destroy_code;
}
#endif
#endif
#endif
