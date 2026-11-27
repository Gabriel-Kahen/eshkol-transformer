#define main mod5_training_smoke_main
#include "../mod5/test_mod5_training_smoke.c"
#undef main

#include "eshkol_transformer/mp4_loss_scaler_record.h"

#pragma STDC FP_CONTRACT OFF

enum {
  TILE = 8,
  ACCEPTED_STEPS = 4,
  GRAD_CLEAN = 0,
  GRAD_NARROW_REJECT = 1,
  GRAD_UNSCALE_OVERFLOW = 2,
  GRAD_F32_SCALE_OVERFLOW = 3
};
_Static_assert((WIDTH * WIDTH) % TILE == 0,
               "parameter rows must fit MP4 tiles");
static const uint64_t tile_shape[] = {TILE};
static const float learning_rate = 0.25f;

typedef struct format {
  const char *dtype, *cast, *unscale;
  float max_scale, desired_raw_fault, finite_narrow_limit, atol, rtol;
} format;

static const format formats[] = {
    {"f16", "cast.f32-to-f16", "grad.unscale-f16-to-f32", 32.0f, 2200.0f,
     65504.0f, 2e-5f, 0.005f},
    {"bf16", "cast.f32-to-bf16", "grad.unscale-bf16-to-f32", 0x1p127f,
     0x1.ff8p0f, 0x1.ffp127f, 1e-4f, 0.02f},
};

static const et_kernel_provider_v1 *resolve_mp4(void *context,
                                                const char *symbol) {
  (void)context;
  return strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0
             ? et_mp4_kernel_provider_v1()
             : NULL;
}

static et_kernel_tensor_view_v1 mp_view(void *data, const char *dtype,
                                        size_t rank, const uint64_t *shape) {
  const size_t width = strcmp(dtype, "f32") == 0
                           ? sizeof(float)
                           : strcmp(dtype, "bool") == 0 ? 1u : sizeof(uint16_t);
  return (et_kernel_tensor_view_v1){sizeof(et_kernel_tensor_view_v1),
                                     data,
                                     rank == 0 ? width : TILE * width,
                                     dtype,
                                     "cpu",
                                     ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,
                                     0u,
                                     rank,
                                     shape};
}

static int cast_tile(const et_kernel_runtime *runtime, const format *fmt,
                     float *scaled, uint16_t *narrow) {
  const et_kernel_request_v1 request = {sizeof(request), fmt->cast, "f32", "cpu",
                                        1u, tile_shape, 1u, {0}};
  const et_kernel_tensor_view_v1 in =
      mp_view(scaled, "f32", 1u, tile_shape);
  et_kernel_tensor_view_v1 out =
      mp_view(narrow, fmt->dtype, 1u, tile_shape);
  const et_kernel_call_v1 call =
      test_call(strcmp(fmt->dtype, "f16") == 0 ? "kernel.cast-f16"
                                                 : "kernel.cast-bf16",
                &request, 1u, &in, 1u, &out);
  et_kernel_error error;
  const int status = et_kernel_runtime_dispatch(runtime, &call, &error);
  if (status != 0)
    CHECK(error.code == ET_KERNEL_CODE_PROVIDER_REJECTED);
  return status;
}

static uint8_t unscale_tile(const et_kernel_runtime *runtime, const format *fmt,
                            uint16_t *narrow, float inverse, float *gradient) {
  const et_kernel_request_v1 request = {sizeof(request), fmt->unscale, "f32",
                                        "cpu", 1u, tile_shape, 1u, {0}};
  const et_kernel_tensor_view_v1 in[] = {
      mp_view(narrow, fmt->dtype, 1u, tile_shape),
      mp_view(&inverse, "f32", 0u, NULL)};
  uint8_t overflow = UINT8_MAX;
  et_kernel_tensor_view_v1 out[] = {
      mp_view(gradient, "f32", 1u, tile_shape),
      mp_view(&overflow, "bool", 0u, NULL)};
  const et_kernel_call_v1 call =
      test_call("kernel.grad-unscale", &request, 2u, in, 2u, out);
  dispatch(runtime, &call);
  CHECK(overflow <= 1u);
  return overflow;
}

typedef struct trajectory {
  weights master;
  et_mp4_loss_scaler_v1 scaler;
} trajectory;

typedef struct step_result {
  weights scaled, unscaled;
  uint16_t narrow[PARAMETERS][WIDTH * WIDTH];
  double loss;
  float fault_peak_scaled;
  uint8_t gradient_status;
  et_mp4_loss_scaler_status scaler_status;
} step_result;

static uint8_t store_and_unscale(const et_kernel_runtime *runtime,
                                 const format *fmt, float scale,
                                 const weights *raw, step_result *result) {
  double activity = 0.0;
  for (size_t p = 0; p < PARAMETERS; ++p)
    for (size_t i = 0; i < param_size(p, 2u); ++i) {
      CHECK(isfinite(raw->data[p][i]));
      activity += fabs(raw->data[p][i]);
    }
  CHECK(activity > 0.001);
  for (size_t p = 0; p < PARAMETERS; ++p) {
    for (size_t offset = 0; offset < WIDTH * WIDTH; offset += TILE) {
      float *scaled = &result->scaled.data[p][offset];
      uint16_t *narrow = &result->narrow[p][offset];
      for (size_t i = 0; i < TILE; ++i) {
        const float source = raw->data[p][offset + i];
        CHECK(isfinite(source));
        scaled[i] = source * scale;
        if (!isfinite(scaled[i]))
          return GRAD_F32_SCALE_OVERFLOW;
      }
      if (cast_tile(runtime, fmt, scaled, narrow) != 0)
        return GRAD_NARROW_REJECT;
      if (unscale_tile(runtime, fmt, narrow, 1.0f / scale,
                       &result->unscaled.data[p][offset]) != 0)
        return GRAD_UNSCALE_OVERFLOW;
    }
  }
  return GRAD_CLEAN;
}

static void apply_sgd(weights *model, const weights *gradient) {
  for (size_t p = 0; p < PARAMETERS; ++p)
    for (size_t i = 0; i < param_size(p, 2u); ++i)
      model->data[p][i] -= learning_rate * gradient->data[p][i];
}

static void check_parity(const weights *actual, const weights *baseline,
                         const format *fmt) {
  for (size_t p = 0; p < PARAMETERS; ++p)
    for (size_t i = 0; i < param_size(p, 2u); ++i)
      check_close(actual->data[p][i], baseline->data[p][i], fmt->atol,
                  fmt->rtol);
}

static void initialize(weights *w, float *x, float *target) {
  memset(w, 0, sizeof(*w));
  for (size_t p = 0; p < PARAMETERS; ++p)
    for (size_t i = 0; i < param_size(p, 2u); ++i) {
      const int seed = (int)((p * 29u + i * 11u + i / 5u) % 37u) - 18;
      w->data[p][i] = (p >= GAMMA1 ? 1.0f : 0.0f) + (float)seed * 0x1p-7f;
    }
  for (size_t i = 0; i < ELEMENTS; ++i) {
    x[i] = (float)((int)((i * 7u + i / 5u) % 31u) - 15) * 0x1p-5f;
    target[i] = 0.65f * x[i] + (float)((int)((i * 3u) % 13u) - 6) * 0x1p-6f;
  }
}

static float max_gradient(const et_kernel_runtime *mod5, const weights *model,
                          float *x, const float *target) {
  weights copy = *model, gradient;
  float dx[ELEMENTS];
  CHECK(isfinite(block_loss(mod5, 2u, &copy, x, target, &gradient, dx)));
  float largest = 0.0f;
  for (size_t p = 0; p < PARAMETERS; ++p)
    for (size_t i = 0; i < param_size(p, 2u); ++i) {
      CHECK(isfinite(gradient.data[p][i]));
      largest = fmaxf(largest, fabsf(gradient.data[p][i]));
    }
  return largest;
}

static void make_fault_target(const et_kernel_runtime *mod5,
                              const weights *model, float *x,
                              const float *normal, float *fault,
                              const format *fmt) {
  float low = 0.0f, high = 1.0f;
  CHECK(max_gradient(mod5, model, x, normal) < fmt->desired_raw_fault);
  for (unsigned expansion = 0; expansion < 32u; ++expansion) {
    for (size_t i = 0; i < ELEMENTS; ++i)
      fault[i] = normal[i] + high;
    if (max_gradient(mod5, model, x, fault) >= fmt->desired_raw_fault)
      break;
    low = high;
    high *= 2.0f;
    CHECK(isfinite(high) && expansion < 31u);
  }
  for (unsigned i = 0; i < 32u; ++i) {
    const float middle = (low + high) * 0.5f;
    if (middle == low || middle == high)
      break;
    for (size_t j = 0; j < ELEMENTS; ++j)
      fault[j] = normal[j] + middle;
    if (max_gradient(mod5, model, x, fault) < fmt->desired_raw_fault)
      low = middle;
    else
      high = middle;
  }
  for (size_t i = 0; i < ELEMENTS; ++i)
    fault[i] = normal[i] + high;
}

static et_mp4_loss_scaler_v1 new_scaler(const format *fmt) {
  et_mp4_loss_scaler_v1 scaler;
  CHECK(et_mp4_loss_scaler_init(&scaler, fmt->max_scale * 0.5f, 2.0f, 0.5f, 2u,
                                1.0f, fmt->max_scale) ==
        ET_MP4_LOSS_SCALER_STEP_ACCEPTED);
  return scaler;
}

static void run_step(const et_kernel_runtime *mod5,
                     const et_kernel_runtime *mp4, const format *fmt,
                     trajectory *state, float *x, const float *target,
                     uint8_t fault, step_result *result) {
  const weights before = state->master;
  weights raw;
  float dx[ELEMENTS];
  result->loss = block_loss(mod5, 2u, &state->master, x, target, &raw, dx);
  CHECK(isfinite(result->loss));
  if (fault) {
    for (size_t p = 0; p < PARAMETERS; ++p)
      for (size_t i = 0; i < param_size(p, 2u); ++i) {
        const float scaled = raw.data[p][i] * state->scaler.scale;
        CHECK(isfinite(raw.data[p][i]) && isfinite(scaled));
        result->fault_peak_scaled =
            fmaxf(result->fault_peak_scaled, fabsf(scaled));
      }
    CHECK(result->fault_peak_scaled > fmt->finite_narrow_limit);
  }
  result->gradient_status =
      store_and_unscale(mp4, fmt, state->scaler.scale, &raw, result);
  result->scaler_status = et_mp4_loss_scaler_update(
      &state->scaler, result->gradient_status != GRAD_CLEAN);
  if (result->scaler_status == ET_MP4_LOSS_SCALER_STEP_ACCEPTED) {
    CHECK(result->gradient_status == GRAD_CLEAN);
    apply_sgd(&state->master, &result->unscaled);
    CHECK(memcmp(&state->master, &before, sizeof(before)) != 0);
  } else {
    CHECK(memcmp(&state->master, &before, sizeof(before)) == 0);
  }
}

static void check_record_parity(const et_mp4_loss_scaler_v1 *control,
                                const et_mp4_loss_scaler_v1 *replay) {
  uint8_t control_record[ET_MP4_LOSS_SCALER_RECORD_V1_BYTES];
  uint8_t replay_record[ET_MP4_LOSS_SCALER_RECORD_V1_BYTES];
  CHECK(et_mp4_loss_scaler_encode_v1(control, control_record,
                                     sizeof(control_record)) ==
        ET_MP4_LOSS_SCALER_RECORD_OK);
  CHECK(et_mp4_loss_scaler_encode_v1(replay, replay_record,
                                     sizeof(replay_record)) ==
        ET_MP4_LOSS_SCALER_RECORD_OK);
  CHECK(memcmp(control_record, replay_record, sizeof(control_record)) == 0);
}

static void run_format(const et_kernel_runtime *mod5,
                       const et_kernel_runtime *mp4, const format *fmt) {
  static const struct {
    uint8_t fault, at_max_scale;
    uint32_t clean_steps;
  } steps[] = {
      {0u, 0u, 1u}, {0u, 1u, 0u}, {1u, 0u, 0u}, {0u, 0u, 1u}, {0u, 1u, 0u}};
  weights baseline;
  float x[ELEMENTS], target[ELEMENTS], fault_target[ELEMENTS], dx[ELEMENTS];
  initialize(&baseline, x, target);
  trajectory control = {baseline, new_scaler(fmt)};
  trajectory prefix = {baseline, new_scaler(fmt)};
  trajectory resumed = {0};
  trajectory *replay = &prefix;
  const float half_scale = fmt->max_scale * 0.5f;

  unsigned accepted = 0;
  float fault_peak_scaled = 0.0f;
  for (size_t attempt = 0; attempt < ARRAY_COUNT(steps); ++attempt) {
    if (attempt == 1u) {
      /* Growth is due on the next accepted update. */
      uint8_t record[ET_MP4_LOSS_SCALER_RECORD_V1_BYTES];
      CHECK(prefix.scaler.scale == half_scale &&
            prefix.scaler.clean_steps == 1u);
      CHECK(et_mp4_loss_scaler_encode_v1(&prefix.scaler, record,
                                         sizeof(record)) ==
            ET_MP4_LOSS_SCALER_RECORD_OK);
      memcpy(&resumed.master, &prefix.master, sizeof(resumed.master));
      CHECK(et_mp4_loss_scaler_decode_v1(record, sizeof(record),
                                         &resumed.scaler) ==
            ET_MP4_LOSS_SCALER_RECORD_OK);
      check_record_parity(&control.scaler, &resumed.scaler);
      replay = &resumed;
    }

    const uint8_t fault = steps[attempt].fault;
    if (fault)
      make_fault_target(mod5, &control.master, x, target, fault_target, fmt);
    const float *step_target = fault ? fault_target : target;
    step_result expected = {0}, actual = {0};
    run_step(mod5, mp4, fmt, &control, x, step_target, fault, &expected);
    run_step(mod5, mp4, fmt, replay, x, step_target, fault, &actual);
    CHECK(actual.gradient_status == expected.gradient_status);
    CHECK(actual.scaler_status == expected.scaler_status);
    CHECK(memcmp(&actual.loss, &expected.loss, sizeof(actual.loss)) == 0);
    CHECK(memcmp(&actual.scaled, &expected.scaled, sizeof(actual.scaled)) == 0);
    CHECK(memcmp(actual.narrow, expected.narrow, sizeof(actual.narrow)) == 0);
    CHECK(memcmp(&actual.unscaled, &expected.unscaled,
                 sizeof(actual.unscaled)) == 0);
    CHECK(memcmp(&replay->master, &control.master, sizeof(control.master)) ==
          0);
    CHECK(control.scaler.scale ==
          (steps[attempt].at_max_scale ? fmt->max_scale : half_scale));
    CHECK(control.scaler.clean_steps == steps[attempt].clean_steps);
    check_record_parity(&control.scaler, &replay->scaler);
    if (fault) {
      CHECK(expected.gradient_status == GRAD_NARROW_REJECT &&
            expected.scaler_status == ET_MP4_LOSS_SCALER_OVERFLOW_BACKOFF);
      CHECK(actual.fault_peak_scaled == expected.fault_peak_scaled);
      fault_peak_scaled = expected.fault_peak_scaled;
      continue;
    }
    CHECK(expected.gradient_status == GRAD_CLEAN &&
          expected.scaler_status == ET_MP4_LOSS_SCALER_STEP_ACCEPTED);
    weights reference_gradient;
    CHECK(isfinite(block_loss(mod5, 2u, &baseline, x, target,
                              &reference_gradient, dx)));
    for (size_t p = 0; p < PARAMETERS; ++p)
      for (size_t i = 0; i < param_size(p, 2u); ++i)
        check_close(expected.unscaled.data[p][i], reference_gradient.data[p][i],
                    fmt->atol, fmt->rtol);
    apply_sgd(&baseline, &reference_gradient);
    check_parity(&control.master, &baseline, fmt);
    ++accepted;
  }
  CHECK(accepted == ACCEPTED_STEPS && replay == &resumed);
  const double control_loss =
      block_loss(mod5, 2u, &control.master, x, target, NULL, NULL);
  const double resumed_loss =
      block_loss(mod5, 2u, &resumed.master, x, target, NULL, NULL);
  CHECK(isfinite(control_loss) &&
        memcmp(&control_loss, &resumed_loss, sizeof(control_loss)) == 0);
  printf("MP4/MOD5 %s real-gradient storage and scaler record replay: PASS "
         "(%u updates, finite scaled cast rejection at %.9g, growth, backoff, "
         "recovery, final loss %.9g)\n",
         fmt->dtype, accepted, fault_peak_scaled, control_loss);
}

#ifndef ET_MP4_MOD5_TEST_ENTRY
#define ET_MP4_MOD5_TEST_ENTRY main
#endif
int ET_MP4_MOD5_TEST_ENTRY(void) {
  et_kernel_runtime *mod5 = discover_mod5();
  et_kernel_runtime *mp4 = NULL;
  et_kernel_error error;
  CHECK(et_kernel_runtime_discover(resolve_mp4, NULL, &mp4, &error) == 0);
  for (size_t i = 0; i < ARRAY_COUNT(formats); ++i)
    run_format(mod5, mp4, &formats[i]);
  et_kernel_runtime_destroy(mp4);
  et_kernel_runtime_destroy(mod5);
  return 0;
}
