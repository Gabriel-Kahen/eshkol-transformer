#define ET_MP4_MOD5_TEST_ENTRY mp4_mod5_gradients_main
#include "test_mp4_mod5_gradients.c"
#undef ET_MP4_MOD5_TEST_ENTRY

#include "../mod5/mod5_o2_fixture.h"

enum { KV_HEADS = 2 };
static unsigned plans, updates;

int32_t __real_et_f32_gradient_plan_prepare_v1(
    size_t, const et_f32_gradient_contribution_v1 *, uint32_t,
    et_f32_gradient_plan **, et_f32_tensor_error *);
int32_t __wrap_et_f32_gradient_plan_prepare_v1(
    size_t count, const et_f32_gradient_contribution_v1 *rows, uint32_t weight,
    et_f32_gradient_plan **plan, et_f32_tensor_error *error) {
  ++plans;
  return __real_et_f32_gradient_plan_prepare_v1(count, rows, weight, plan, error);
}

int32_t __real_et_o2_optimizer_step_v1(et_o2_optimizer *, et_o2_error_v1 *);
int32_t __wrap_et_o2_optimizer_step_v1(et_o2_optimizer *optimizer,
                                      et_o2_error_v1 *error) {
  ++updates;
  return __real_et_o2_optimizer_step_v1(optimizer, error);
}

static float peak_scaled(const weights *gradient, float scale) {
  float peak = 0;
  for (size_t p = 0; p < PARAMETERS; ++p) {
    for (size_t i = 0; i < param_size(p, KV_HEADS); ++i) {
      const float scaled = gradient->data[p][i] * scale;
      CHECK(isfinite(gradient->data[p][i]) && isfinite(scaled));
      peak = fmaxf(peak, fabsf(scaled));
    }
  }
  return peak;
}

static void check_moments(const state_bits *state) {
  for (size_t p = 0; p < PARAMETERS; ++p) {
    for (size_t kind = 0; kind < MOMENT_COUNT; ++kind) {
      uint32_t active = 0;
      for (size_t i = 0; i < param_size(p, KV_HEADS); ++i) {
        const uint32_t bits = state->moments[p][kind][i];
        CHECK((bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000));
        CHECK(kind != ET_O2_MOMENT_EXP_AVG_SQ || (bits & UINT32_C(0x80000000)) == 0);
        active |= bits & UINT32_C(0x7fffffff);
      }
      CHECK(active != 0);
    }
  }
}

static void write_bytes(FILE *out, const void *data, size_t bytes) {
  CHECK(fwrite(data, 1, bytes, out) == bytes);
}

static void write_attempt(FILE *out, const weights *raw, const step_result *step,
                          const state_bits *state, const et_mp4_loss_scaler_v1 *scaler) {
  write_bytes(out, raw, sizeof(*raw));
  write_bytes(out, &step->loss, sizeof(step->loss));
  write_bytes(out, &step->fault_peak_scaled, sizeof(step->fault_peak_scaled));
  write_bytes(out, &step->gradient_status, sizeof(step->gradient_status));
  write_bytes(out, &step->scaler_status, sizeof(step->scaler_status));
  write_bytes(out, &step->scaled, sizeof(step->scaled));
  write_bytes(out, &step->narrow, sizeof(step->narrow));
  write_bytes(out, &step->unscaled, sizeof(step->unscaled));
  write_bytes(out, &scaler->scale, sizeof(scaler->scale));
  write_bytes(out, &scaler->clean_steps, sizeof(scaler->clean_steps));
  write_bytes(out, &state->completed, sizeof(state->completed));
  write_bytes(out, &state->schedule_factor, sizeof(state->schedule_factor));
  write_bytes(out, &state->values, sizeof(state->values));
  write_bytes(out, &state->moments, sizeof(state->moments));
  write_bytes(out, &state->options, sizeof(state->options));
}

static void run_adamw(const et_kernel_runtime *mod5, const et_kernel_runtime *mp4,
                      const format *fmt, FILE *trace) {
  static const struct {
    unsigned clean_steps;
    int at_max;
  } expected_scaler[] = {{1, 0}, {0, 1}, {0, 0}, {1, 0}, {0, 1}};
  weights master;
  float x[ELEMENTS], target[ELEMENTS], fault_target[ELEMENTS], dx[ELEMENTS];
  initialize(&master, x, target);
  state_bits initial = {0};
  memcpy(initial.values, master.data, sizeof(initial.values));
  const live_model model = create_model(&initial, KV_HEADS, 0, 0);
  et_mp4_loss_scaler_v1 scaler = new_scaler(fmt);
  const float half = fmt->max_scale * 0.5f;
  weights rejected_gradient;
  double rejected_loss = 0;
  float overflow_peak = 0;

  for (size_t attempt = 0; attempt < ARRAY_COUNT(expected_scaler); ++attempt) {
    const state_bits before = capture(&model, KV_HEADS);
    memcpy(master.data, before.values, sizeof(before.values));
    const int reject = attempt == 2, retry = attempt == 3;
    if (reject)
      make_fault_target(mod5, &master, x, target, fault_target, fmt);
    const float *active_target = reject || retry ? fault_target : target;
    weights gradient;
    step_result result = {0};
    result.loss = block_loss(mod5, KV_HEADS, &master, x, active_target, &gradient, dx);
    CHECK(isfinite(result.loss));
    if (reject || retry) {
      result.fault_peak_scaled = peak_scaled(&gradient, scaler.scale);
      if (reject)
        CHECK(result.fault_peak_scaled > fmt->finite_narrow_limit);
      else
        CHECK(result.fault_peak_scaled < fmt->finite_narrow_limit);
    }
    if (retry) {
      CHECK(memcmp(&result.loss, &rejected_loss, sizeof(rejected_loss)) == 0);
      CHECK(memcmp(&gradient, &rejected_gradient, sizeof(gradient)) == 0);
    }

    const unsigned before_plans = plans, before_updates = updates;
    result.gradient_status = store_and_unscale(mp4, fmt, scaler.scale, &gradient, &result);
    result.scaler_status = et_mp4_loss_scaler_update(&scaler,
                                                    result.gradient_status != GRAD_CLEAN);
    if (result.scaler_status == ET_MP4_LOSS_SCALER_STEP_ACCEPTED) {
      CHECK(result.gradient_status == GRAD_CLEAN);
      check_parity(&result.unscaled, &gradient, fmt);
      apply_o2_gradient(&model, KV_HEADS, &result.unscaled);
    }
    CHECK(et_o2_optimizer_require_absent_gradients_v1(model.optimizer, NULL) == 0);
    const state_bits after = capture(&model, KV_HEADS);
    check_moments(&after);
    if (reject) {
      CHECK(result.gradient_status == GRAD_NARROW_REJECT);
      CHECK(result.scaler_status == ET_MP4_LOSS_SCALER_OVERFLOW_BACKOFF);
      CHECK(plans == before_plans && updates == before_updates);
      CHECK(before.completed == 2);
      check_same_state(&after, &before);
      rejected_loss = result.loss;
      rejected_gradient = gradient;
      overflow_peak = result.fault_peak_scaled;
    } else {
      CHECK(result.scaler_status == ET_MP4_LOSS_SCALER_STEP_ACCEPTED);
      CHECK(plans == before_plans + 1 && updates == before_updates + 1);
      CHECK(after.completed == before.completed + 1);
      CHECK(memcmp(after.values, before.values, sizeof(before.values)) != 0);
    }
    CHECK(scaler.scale == (expected_scaler[attempt].at_max ? fmt->max_scale : half));
    CHECK(scaler.clean_steps == expected_scaler[attempt].clean_steps);
    write_attempt(trace, &gradient, &result, &after, &scaler);
  }
  CHECK(plans == ACCEPTED_STEPS && updates == ACCEPTED_STEPS);
  printf("MP4/MOD5/O2 %s PASS: %u AdamW updates, finite narrowing rejection at %.9g, "
         "exact skipped state, same-target retry and scaler regrowth\n",
         fmt->dtype, updates, (double)overflow_peak);
}

int main(int argc, char **argv) {
  CHECK(argc == 3);
  const format *fmt = NULL;
  for (size_t i = 0; i < ARRAY_COUNT(formats); ++i)
    if (strcmp(argv[1], formats[i].dtype) == 0)
      fmt = &formats[i];
  CHECK(fmt != NULL);
  FILE *trace = fopen(argv[2], "wb");
  CHECK(trace != NULL);
  et_kernel_runtime *mod5 = discover_mod5(), *mp4 = NULL;
  et_kernel_error error;
  CHECK(et_kernel_runtime_discover(resolve_mp4, NULL, &mp4, &error) == 0);
  run_adamw(mod5, mp4, fmt, trace);
  CHECK(fclose(trace) == 0);
  et_kernel_runtime_destroy(mp4);
  et_kernel_runtime_destroy(mod5);
  return EXIT_SUCCESS;
}
