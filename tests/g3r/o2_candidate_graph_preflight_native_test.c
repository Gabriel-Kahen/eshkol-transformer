#define ET_G3R_CANDIDATE_RETIRE_PRIVATE 1
#define ET_G3R_O2_GRAPH_PREFLIGHT_PRIVATE 1
#include "../../src/eshkol_transformer/m3t_transport.c"

#include <stdio.h>

static int checks;
#define CHECK(value) do { \
  ++checks; \
  if (!(value)) { \
    fprintf(stderr, "O2 graph preflight failed at line %d\n", __LINE__); \
    return 1; \
  } \
} while (0)

static int make_owner(owner **result, unsigned char handles[14],
                      initializer *seed) {
  owner *candidate = et_m3t_private_owner_create_v1(seed);
  CHECK(candidate != NULL);
  for (int64_t index = 0; index < 14; ++index)
    CHECK(et_m3t_private_owner_bind_v1(candidate, index,
                                       &handles[index]) == 0);
  for (int64_t index = 0; index < 14; ++index)
    CHECK(et_m3t_private_owner_initialize_v1(candidate, index) == 0);
  CHECK(et_m3t_private_owner_seal_v1(candidate) == 0);
  CHECK(et_m3t_private_initializer_seal_v1(candidate->successor) == 0);
  *result = candidate;
  return 0;
}

static int make_optimizer(et_o2_optimizer **result, owner *candidate) {
  et_o2_optimizer_builder *builder = NULL;
  et_o2_error_v1 error = {0};
  CHECK(et_o2_optimizer_builder_create_v1(
      14u, ET_O2_CLIP_NONE, 0u, ET_O2_SCHEDULE_CONSTANT, 0u, 0u,
      UINT32_C(0x3f800000), &builder, &error) == 0);
  for (size_t index = 0; index < 14u; ++index)
    CHECK(et_o2_optimizer_builder_set_v1(
        builder, index, candidate->p[index], candidate->handles[index],
        UINT32_C(0x3ca3d70a), UINT32_C(0x3f4ccccd),
        UINT32_C(0x3f7d70a4), UINT32_C(0x322bcc77), 0u, &error) == 0);
  CHECK(et_o2_optimizer_builder_finish_v1(&builder, result, &error) == 0);
  return 0;
}

int main(void) {
  unsigned char handles[14] = {0};
  unsigned char foreign_handles[14] = {0};
  initializer *seed = et_m3t_private_initializer_create_v1(1729);
  CHECK(seed != NULL);
  CHECK(et_m3t_private_initializer_seal_v1(seed) == 0);
  owner *exact = NULL;
  CHECK(make_owner(&exact, handles, seed) == 0);
  CHECK(et_m3t_private_candidate_o2_graph_preflight_v1(
            exact, NULL) == ET_M3T_INVALID_STATE);
  et_o2_optimizer *optimizer = NULL;
  CHECK(make_optimizer(&optimizer, exact) == 0);
  CHECK(et_m3t_private_candidate_o2_graph_preflight_v1(
            exact, optimizer) == 0);
  CHECK(et_m3t_private_candidate_o2_graph_preflight_v1(
            NULL, optimizer) == ET_M3T_INVALID_ARGUMENT);
  CHECK(et_m3t_private_candidate_o2_graph_preflight_v1(
            seed, optimizer) == ET_M3T_INVALID_ARGUMENT);
  et_o2_error_v1 error = {0};
  CHECK(et_o2_private_candidate_graph_entry_match_v1(
      optimizer, 14u, exact->p[0], exact->handles[0],
      et_f32_parameter_canonical_owner_v1(exact->p[0]),
      &error) == ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(et_o2_private_candidate_graph_entry_match_v1(
      optimizer, 0u, exact->p[1], exact->handles[0],
      et_f32_parameter_canonical_owner_v1(exact->p[0]),
      &error) == ET_O2_STATUS_INVALID_STATE);
  CHECK(et_o2_private_candidate_graph_entry_match_v1(
      optimizer, 0u, exact->p[0], exact->handles[1],
      et_f32_parameter_canonical_owner_v1(exact->p[0]),
      &error) == ET_O2_STATUS_INVALID_STATE);
  CHECK(et_o2_private_candidate_graph_entry_match_v1(
      optimizer, 0u, exact->p[0], exact->handles[0],
      et_f32_parameter_canonical_owner_v1(exact->p[1]),
      &error) == ET_O2_STATUS_INVALID_STATE);
  void *saved_handle = exact->handles[0];
  exact->handles[0] = exact->handles[1];
  CHECK(et_m3t_private_candidate_o2_graph_preflight_v1(
            exact, optimizer) == ET_M3T_INVALID_STATE);
  exact->handles[0] = saved_handle;
  et_f32_parameter *saved_parameter = exact->p[0];
  exact->p[0] = exact->p[1];
  CHECK(et_m3t_private_candidate_o2_graph_preflight_v1(
            exact, optimizer) == ET_M3T_INVALID_STATE);
  exact->p[0] = saved_parameter;
  CHECK(et_m3t_private_candidate_o2_graph_preflight_v1(
            exact, optimizer) == 0);
  owner *foreign = NULL;
  CHECK(make_owner(&foreign, foreign_handles, seed) == 0);
  et_o2_optimizer *other = NULL;
  CHECK(make_optimizer(&other, foreign) == 0);
  CHECK(et_m3t_private_candidate_o2_graph_preflight_v1(
            exact, other) == ET_M3T_INVALID_STATE);
  CHECK(et_m3t_private_candidate_o2_graph_preflight_v1(
            foreign, optimizer) == ET_M3T_INVALID_STATE);
  CHECK(et_m3t_private_candidate_o2_graph_preflight_v1(
            exact, optimizer) == 0);
  et_o2_optimizer_state *state = NULL;
  CHECK(et_o2_optimizer_state_snapshot_v1(optimizer, &state, &error) == 0);
  CHECK(et_m3t_private_candidate_o2_graph_preflight_v1(
            exact, optimizer) == ET_M3T_INVALID_STATE);
  CHECK(et_o2_optimizer_state_release_v1(state, &error) == 0);
  CHECK(et_m3t_private_candidate_o2_graph_preflight_v1(
            exact, optimizer) == 0);
  owner *stale = et_m3t_private_owner_create_v1(seed);
  CHECK(stale != NULL);
  CHECK(et_m3t_private_owner_abort_v1(stale) == 0);
  CHECK(et_m3t_private_candidate_o2_graph_preflight_v1(
            stale, optimizer) == ET_M3T_INVALID_STATE);
  CHECK(et_o2_private_candidate_retire_commit_v1(optimizer, &error) == 0);
  CHECK(et_m3t_private_candidate_o2_graph_preflight_v1(
            exact, optimizer) == ET_M3T_INVALID_STATE);
  puts("G3-R native O2 graph identity PASS");
  printf("checks=%d\n", checks);
  return 0;
}
