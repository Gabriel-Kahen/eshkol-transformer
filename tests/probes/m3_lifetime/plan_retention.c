/* Non-freezing development probe of existing I2 only. Including the production
 * source permits layout assertions; operations below use its existing APIs.
 * Local build: cc -std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
 * -fexcess-precision=standard -frounding-math -Iinclude -Inative
 * tests/probes/m3_lifetime/plan_retention.c native/kernel_abi.c -o /tmp/m3-plan
 */
#ifndef ET_F32_TENSOR_TESTING
#define ET_F32_TENSOR_TESTING 1
#endif
#include "../../../native/f32_tensor.c"

_Static_assert(sizeof(et_f32_tensor) == 88, "x86-64 I2 tensor shell");
_Static_assert(sizeof(et_f32_gradient_plan) == 48, "x86-64 gradient plan");
_Static_assert(sizeof(et_f32_gradient_reset_plan) == 40, "x86-64 reset plan");

#define CHECK(expr) do { if (!(expr)) { \
  fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #expr); exit(1); \
} } while (0)

static et_f32_tensor_error error;

static et_f32_test_live_counts_v1 live_counts(void) {
  et_f32_test_live_counts_v1 out = {.struct_size = sizeof(out)};
  et_f32_test_live_counts_snapshot_v1(&out);
  return out;
}

static et_f32_test_retired_counts_v1 retired_counts(void) {
  et_f32_test_retired_counts_v1 out = {.struct_size = sizeof(out)};
  et_f32_test_retired_counts_snapshot_v1(&out);
  return out;
}

static void same_live(et_f32_test_live_counts_v1 a,
                      et_f32_test_live_counts_v1 b) {
  CHECK(a.tensors == b.tensors && a.parameters == b.parameters);
  CHECK(a.borrows == b.borrows && a.copy_plans == b.copy_plans);
  CHECK(a.gradient_plans == b.gradient_plans && a.reset_plans == b.reset_plans);
  CHECK(a.owned_clones == b.owned_clones);
}

static void metadata(et_f32_parameter *p, uint32_t state,
                     uint64_t count, uint32_t weight) {
  et_f32_gradient_metadata_v1 m = {.struct_size = sizeof(m)};
  int32_t zero = 0, finite = 0;
  CHECK(et_f32_parameter_gradient_metadata_v1(p, &m, &error) == 0);
  CHECK(m.state == state && m.contribution_count == count);
  CHECK(m.normalization_weight_bits == weight);
  if (state == ET_F32_GRADIENT_ABSENT) {
    /* Inspection rejects absent; the next successful prepare validates its
     * exact-positive-zero backing through existing I2 admission. */
    CHECK(et_f32_parameter_gradient_exact_positive_zero_v1(p, &zero, &error) ==
          ET_F32_TENSOR_ERROR_INVALID_STATE);
    CHECK(et_f32_parameter_gradient_finite_v1(p, &finite, &error) ==
          ET_F32_TENSOR_ERROR_INVALID_STATE);
    CHECK(zero == 0 && finite == 0);
    return;
  }
  CHECK(et_f32_parameter_gradient_exact_positive_zero_v1(p, &zero, &error) == 0);
  CHECK(et_f32_parameter_gradient_finite_v1(p, &finite, &error) == 0);
  CHECK(zero == 1 && finite == 1);
}

static void contribute(et_f32_parameter *p, et_f32_tensor *source,
                       uint64_t ordinal, uint32_t weight) {
  et_f32_gradient_contribution_v1 row = {
    .struct_size = sizeof(row), .destination = p,
    .weighted_numerator = source, .expected_ordinal = ordinal
  };
  et_f32_gradient_plan *plan = NULL;
  CHECK(et_f32_gradient_plan_prepare_v1(1, &row, weight, &plan, &error) == 0);
  CHECK(live_counts().gradient_plans == 1);
  et_f32_tensor_test_fail_alloc_after_v1(0);
  CHECK(et_f32_gradient_plan_commit_v1(plan, &error) == 0);
  CHECK(et_f32_gradient_plan_commit_v1(plan, &error) != 0);
  CHECK(et_f32_gradient_plan_release_v1(&plan, &error) == 0);
  CHECK(plan == NULL && live_counts().gradient_plans == 0);
  CHECK(et_f32_gradient_plan_release_v1(&plan, &error) == 0);
  et_f32_tensor_test_reset_allocator_v1();
}

static void reset(et_f32_parameter *p) {
  et_f32_gradient_reset_plan *plan = NULL;
  CHECK(et_f32_gradient_reset_plan_prepare_v1(1, &p, &plan, &error) == 0);
  CHECK(live_counts().reset_plans == 1);
  et_f32_tensor_test_fail_alloc_after_v1(0);
  CHECK(et_f32_gradient_reset_plan_commit_v1(plan, &error) == 0);
  CHECK(et_f32_gradient_reset_plan_commit_v1(plan, &error) != 0);
  CHECK(et_f32_gradient_reset_plan_release_v1(&plan, &error) == 0);
  CHECK(plan == NULL && live_counts().reset_plans == 0);
  CHECK(et_f32_gradient_reset_plan_release_v1(&plan, &error) == 0);
  et_f32_tensor_test_reset_allocator_v1();
}

static void rejects_preserve_present_zero(et_f32_parameter *p,
                                          et_f32_tensor *source) {
  et_f32_test_live_counts_v1 live = live_counts();
  et_f32_test_retired_counts_v1 retired = retired_counts();
  et_f32_gradient_contribution_v1 row = {
    .struct_size = sizeof(row), .destination = p,
    .weighted_numerator = source, .expected_ordinal = 2
  };
  et_f32_gradient_plan *plan = NULL;
  et_f32_gradient_reset_plan *reset_plan = NULL;
  /* One parameter requires plan + entry + numerator scratch allocations. */
  for (size_t allowed = 0; allowed < 3; ++allowed) {
    et_f32_tensor_test_fail_alloc_after_v1(allowed);
    CHECK(et_f32_gradient_plan_prepare_v1(1, &row, 0x3f800000u,
                                         &plan, &error) != 0);
    CHECK(error.code == ET_F32_TENSOR_CODE_ALLOCATION_FAILED && plan == NULL);
    et_f32_tensor_test_reset_allocator_v1();
    metadata(p, ET_F32_GRADIENT_PRESENT, 2, 0x3fc00000u);
    same_live(live, live_counts());
  }
  et_f32_tensor_test_fail_alloc_after_v1(0);
  CHECK(et_f32_gradient_reset_plan_prepare_v1(1, &p, &reset_plan, &error) != 0);
  CHECK(error.code == ET_F32_TENSOR_CODE_ALLOCATION_FAILED && reset_plan == NULL);
  et_f32_tensor_test_reset_allocator_v1();
  row.expected_ordinal = 1;
  CHECK(et_f32_gradient_plan_prepare_v1(1, &row, 0x3f800000u,
                                       &plan, &error) != 0);
  row.expected_ordinal = 2;
  CHECK(et_f32_gradient_plan_prepare_v1(1, &row, 0, &plan, &error) != 0);
  CHECK(plan == NULL);
  metadata(p, ET_F32_GRADIENT_PRESENT, 2, 0x3fc00000u);
  same_live(live, live_counts());
  CHECK(retired_counts().retained_control_bytes == retired.retained_control_bytes);
}

int main(void) {
  const uint64_t shape[] = {4};
  const uint32_t values[] = {0x3f800000u, 0xbf800000u, 0x40000000u, 0};
  uint32_t actual[4];
  const uint32_t zero_bits[4] = {0};
  size_t value_bytes = 0, source_bytes = 0, gradient_bytes = 0;
  int identity = 0;
  et_f32_tensor *initial = NULL, *source = NULL;
  et_f32_parameter *parameter = NULL;
  const et_f32_tensor *value = NULL;
  et_f32_test_live_counts_v1 origin = live_counts();
  CHECK(et_f32_tensor_create_v1(1, shape, &initial, &error) == 0);
  CHECK(et_f32_tensor_copy_bits_from_v1(initial, values, 4, &error) == 0);
  CHECK(et_f32_parameter_create_v1(initial, &parameter, &error) == 0);
  CHECK(et_f32_parameter_bind_identity_v1(parameter, &identity, &error) == 0);
  CHECK(et_f32_parameter_validate_identity_v1(parameter, &identity, &error) == 0);
  CHECK(et_f32_tensor_destroy_v1(&initial, &error) == 0);
  CHECK(et_f32_tensor_create_v1(1, shape, &source, &error) == 0);
  CHECK(et_f32_parameter_value_tensor_v1(parameter, &value, &error) == 0);
  metadata(parameter, ET_F32_GRADIENT_ABSENT, 0, 0);
  contribute(parameter, source, 0, 0x3f800000u);
  metadata(parameter, ET_F32_GRADIENT_PRESENT, 1, 0x3f800000u);
  contribute(parameter, source, 1, 0x3f000000u);
  metadata(parameter, ET_F32_GRADIENT_PRESENT, 2, 0x3fc00000u);
  rejects_preserve_present_zero(parameter, source);
  /* Establish each fixed carrier's exact byte length before retention baseline. */
  CHECK(et_f32_parameter_gradient_snapshot_v1(parameter, &initial, &error) == 0);
  CHECK(et_f32_tensor_byte_length_v1(initial, &gradient_bytes, &error) == 0);
  CHECK(et_f32_tensor_destroy_v1(&initial, &error) == 0);
  CHECK(et_f32_tensor_byte_length_v1(value, &value_bytes, &error) == 0);
  CHECK(et_f32_tensor_byte_length_v1(source, &source_bytes, &error) == 0);
  CHECK(value_bytes == 16 && source_bytes == 16 && gradient_bytes == 16);
  reset(parameter);
  metadata(parameter, ET_F32_GRADIENT_ABSENT, 0, 0);

  et_f32_test_live_counts_v1 baseline = live_counts();
  et_f32_test_retired_counts_v1 before = retired_counts();
  CHECK(baseline.tensors == 3 && baseline.parameters == 1);
  for (size_t cycle = 1; cycle <= 1000; ++cycle) {
    contribute(parameter, source, 0, 0x3f800000u);
    metadata(parameter, ET_F32_GRADIENT_PRESENT, 1, 0x3f800000u);
    reset(parameter);
    metadata(parameter, ET_F32_GRADIENT_ABSENT, 0, 0);
    same_live(baseline, live_counts());
    CHECK(et_f32_tensor_copy_bits_to_v1(value, actual, 4, &error) == 0);
    CHECK(memcmp(actual, values, sizeof(values)) == 0);
    CHECK(et_f32_tensor_copy_bits_to_v1(source, actual, 4, &error) == 0);
    CHECK(memcmp(actual, zero_bits, sizeof(zero_bits)) == 0);
    if (cycle == 10 || cycle == 100 || cycle == 1000) {
      et_f32_test_retired_counts_v1 now = retired_counts();
      CHECK(now.gradient_plans - before.gradient_plans == cycle);
      CHECK(now.reset_plans - before.reset_plans == cycle);
      CHECK(now.retained_control_bytes - before.retained_control_bytes == 88 * cycle);
      CHECK(now.tensors == before.tensors && now.parameters == before.parameters);
      CHECK(now.borrows == before.borrows && now.copy_plans == before.copy_plans);
      printf("cycles=%zu live-tensors=3 live-parameters=1 live-plans=0 "
             "live-tensor-payload=%zu retired-gradient=%zu retired-reset=%zu "
             "retired-control-delta=%zu\n", cycle,
             value_bytes + source_bytes + gradient_bytes,
             now.gradient_plans - before.gradient_plans,
             now.reset_plans - before.reset_plans,
             now.retained_control_bytes - before.retained_control_bytes);
    }
  }
  CHECK(et_f32_tensor_destroy_v1(&source, &error) == 0);
  CHECK(et_f32_parameter_destroy_v1(&parameter, &error) == 0);
  same_live(origin, live_counts());
  puts("PASS existing-I2 retention and zero-gradient metadata; live payload released, "
       "retired controls remain until process exit");
  return 0;
}
