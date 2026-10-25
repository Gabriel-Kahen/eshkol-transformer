#define ET_G3R_CANDIDATE_RETIRE_PRIVATE 1
#include "../../src/eshkol_transformer/m3t_transport.c"
#include <stdio.h>

#define CHECK(condition) do { \
  if (!(condition)) { \
    fprintf(stderr, "binding preflight failed at line %d\n", __LINE__); \
    return 1; \
  } \
  ++checks; \
} while (0)

int main(void) {
  int checks = 0;
  unsigned char handles[15] = {0};
  initializer *seed = et_m3t_private_initializer_create_v1(1729);
  CHECK(seed != NULL);
  CHECK(et_m3t_private_initializer_seal_v1(seed) == 0);
  owner *exact = et_m3t_private_owner_create_v1(seed);
  CHECK(exact != NULL);
  CHECK(et_m3t_private_candidate_binding_preflight_v1(
            exact, 0, &handles[0]) == ET_M3T_INVALID_STATE);
  for (int64_t i = 0; i < 14; ++i)
    CHECK(et_m3t_private_owner_bind_v1(exact, i, &handles[i]) == 0);
  for (int64_t i = 0; i < 14; ++i)
    CHECK(et_m3t_private_owner_initialize_v1(exact, i) == 0);
  CHECK(et_m3t_private_owner_seal_v1(exact) == 0);
  CHECK(et_m3t_private_initializer_seal_v1(exact->successor) == 0);
  for (int64_t i = 0; i < 14; ++i)
    CHECK(et_m3t_private_candidate_binding_preflight_v1(
              exact, i, &handles[i]) == 0);
  CHECK(et_m3t_private_candidate_binding_preflight_v1(
            NULL, 0, &handles[0]) == ET_M3T_INVALID_ARGUMENT);
  CHECK(et_m3t_private_candidate_binding_preflight_v1(
            seed, 0, &handles[0]) == ET_M3T_INVALID_ARGUMENT);
  CHECK(et_m3t_private_candidate_binding_preflight_v1(
            exact, -1, &handles[0]) == ET_M3T_INVALID_ARGUMENT);
  CHECK(et_m3t_private_candidate_binding_preflight_v1(
            exact, 14, &handles[0]) == ET_M3T_INVALID_ARGUMENT);
  CHECK(et_m3t_private_candidate_binding_preflight_v1(
            exact, 0, NULL) == ET_M3T_INVALID_ARGUMENT);
  CHECK(et_m3t_private_candidate_binding_preflight_v1(
            exact, 0, &handles[14]) == ET_M3T_INVALID_STATE);
  void *saved_handle = exact->handles[0];
  exact->handles[0] = exact->handles[1];
  CHECK(et_m3t_private_candidate_binding_preflight_v1(
            exact, 0, &handles[0]) == ET_M3T_INVALID_STATE);
  exact->handles[0] = saved_handle;
  et_f32_parameter *saved_parameter = exact->p[0];
  exact->p[0] = exact->p[1];
  CHECK(et_m3t_private_candidate_binding_preflight_v1(
            exact, 0, &handles[0]) == ET_M3T_INVALID_STATE);
  exact->p[0] = saved_parameter;
  exact->initialized = 13;
  CHECK(et_m3t_private_candidate_binding_preflight_v1(
            exact, 0, &handles[0]) == ET_M3T_INVALID_STATE);
  exact->initialized = 14;
  CHECK(et_m3t_private_candidate_binding_preflight_v1(
            exact, 0, &handles[0]) == 0);
  owner *stale = et_m3t_private_owner_create_v1(seed);
  CHECK(stale != NULL);
  CHECK(et_m3t_private_owner_abort_v1(stale) == 0);
  CHECK(et_m3t_private_candidate_binding_preflight_v1(
            stale, 0, &handles[0]) == ET_M3T_INVALID_STATE);
  puts("G3-R native binding preflight PASS");
  printf("checks=%d\n", checks);
  return 0;
}
